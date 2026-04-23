"""Training pipeline for Static PPR subgraph link prediction.

Architecture: SubgraphClassifier (DRNL + content features → GNN → graph-level MLP).
Loss: K-way cross-entropy with in-subgraph negatives.

Why in-subgraph negatives:
  Global random negatives are structurally unrelated to (u, v) — the model
  trivially learns to tell "PPR-neighborhood of existing edge" apart from
  "PPR-neighborhood of random pair" in a few epochs.

  In-subgraph negatives (w ≠ v, w inside G_{u,v}) are structurally grounded:
  w is already in u's neighbourhood, shares similar PPR mass, and differs only
  in whether the specific edge (u, w) exists.  This forces the model to learn
  edge-specific structural patterns (exactly what DRNL is designed for).

  The GNN is run once per batch; score_pairs() is called K+1 times on the
  cached embeddings — no extra subgraph extraction or cache files needed.
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.subgraph_csr import SubgraphCSR
from ..utils.drnl import compute_drnl_for_batch
from .evaluator import evaluate_ppr_lcilp


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch_lcilp(classifier, pos_cache, optimizer,
                      batch_size, device, x_full=None, drnl_max_dist=6,
                      grad_clip=1.0, k_neg=10,
                      edges_per_epoch=None, verbose=False):
    """One epoch: K-way CE with k_neg in-subgraph negatives per positive.

    The GNN encodes each subgraph once; score_pairs() is then called k_neg+1
    times on the cached embeddings (cheap MLP calls only).
    """
    classifier.train()

    n_total = len(pos_cache)
    if edges_per_epoch and edges_per_epoch < n_total:
        indices = torch.randperm(n_total, device=device)[:edges_per_epoch]
    else:
        indices = torch.randperm(n_total, device=device)

    dataloader = DataLoader(indices.tolist(), batch_size, shuffle=False)
    if verbose:
        dataloader = tqdm(dataloader, desc="  Batches", leave=False,
                          mininterval=10)

    if x_full is None:
        n_nodes = int(pos_cache.node_ids.max().item()) + 1
        x_full = torch.zeros(n_nodes, 1, device=device)

    total_loss, total_examples = 0.0, 0

    for perm in dataloader:
        idx = torch.as_tensor(perm, dtype=torch.long, device=device)
        valid = pos_cache.valid_mask[idx]
        valid_idx = idx[valid]
        if valid_idx.numel() == 0:
            continue

        pos_b = pos_cache.make_batch(valid_idx, x_full)
        B = int(pos_b["u_idx"].size(0))
        if B == 0 or pos_b["total_edges"] == 0:
            continue

        if pos_b["drnl_x"] is not None:
            x = torch.cat([pos_b["drnl_x"].to(device),
                            pos_b["x"].to(device)], dim=-1)
        else:
            x = compute_drnl_for_batch(pos_b, max_dist=drnl_max_dist)

        bvec = torch.repeat_interleave(
            torch.arange(B, device=device), pos_b["num_nodes_vec"])

        optimizer.zero_grad()

        # Encode once — share h and g across all pair scorings
        h, g = classifier.encode(x, pos_b["edge_index"], bvec)

        score_pos = classifier.score_pairs(h, g, pos_b["u_idx"], pos_b["v_idx"])

        # In-subgraph negative sampling: k_neg random nodes ≠ v per subgraph
        offsets = pos_b["batch_node_offsets"][:B]  # [B] start offset of each subgraph
        n_nodes = pos_b["num_nodes_vec"]            # [B] nodes per subgraph
        v_local = pos_b["v_idx"] - offsets          # [B] local position of v

        neg_scores = []
        for _ in range(k_neg):
            # Sample uniform in [0, n_nodes-2], then shift past v to exclude it
            nn_range = (n_nodes - 1).clamp(min=1).float()
            neg_local = (torch.rand(B, device=device) * nn_range).long()
            neg_local = neg_local.clamp(max=n_nodes - 2)
            neg_local = torch.where(neg_local >= v_local,
                                    neg_local + 1, neg_local)
            neg_local = neg_local.clamp(max=n_nodes - 1)
            neg_idx = offsets + neg_local
            neg_scores.append(classifier.score_pairs(h, g, pos_b["u_idx"], neg_idx))

        # K-way CE: [pos, neg_0, ..., neg_{K-1}] — positive is always index 0
        logits = torch.stack([score_pos] + neg_scores, dim=1)   # [B, k_neg+1]
        target = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * B
        total_examples += B

    return total_loss / max(total_examples, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model_ppr_lcilp(classifier, data, split_edge, ppr_extractor,
                           pos_cache=None,
                           epochs=150, batch_size=256, lr=0.01,
                           eval_steps=5, device="cpu", verbose=True,
                           patience=10, min_delta=0.0001,
                           weight_decay=5e-4, grad_clip=1.0,
                           drnl_max_dist=6, k_neg=10,
                           edges_per_epoch=None, cache_dir=None,
                           max_eval_edges=2000, eval_num_negs=None,
                           **_kwargs):
    """Train SubgraphClassifier with in-subgraph K-way CE ranking.

    Parameters
    ----------
    pos_cache : SubgraphCSR or None
        Pre-built positive subgraph cache.  Built from split_edge['train']
        using ppr_extractor if None.
    k_neg : int
        In-subgraph negatives per positive per training step.
    """
    from .trainer_batched import build_or_load_ppr_csr_cache

    classifier = classifier.to(device)

    source_edge = split_edge["train"]["source_node"]
    target_edge = split_edge["train"]["target_node"]

    # --- positive cache ---
    if pos_cache is None:
        pos_cache = build_or_load_ppr_csr_cache(
            source_edge, target_edge, data, ppr_extractor,
            cache_dir=cache_dir, verbose=verbose,
        )

    # --- pre-compute DRNL once ---
    from ..utils.drnl import compute_drnl_for_csr
    if pos_cache.drnl_feats is None:
        if verbose:
            print(f"[DRNL] Pre-computing DRNL for pos cache "
                  f"({int(pos_cache.valid_mask.sum())} subgraphs)...")
        pos_cache.drnl_feats = compute_drnl_for_csr(
            pos_cache, max_dist=drnl_max_dist, verbose=verbose)
        if cache_dir:
            path = os.path.join(cache_dir, "train_subgraphs_csr.pt")
            pos_cache.save(path)
            if verbose:
                mb = os.path.getsize(path) / 1e6
                print(f"[DRNL] Saved pos cache with DRNL: {path} ({mb:.0f} MB)")

    pos_cache = pos_cache.to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10)

    history = {
        "train_loss": [], "val_results": [], "epoch_times": [],
        "learning_rates": [], "best_val_mrr": 0.0, "best_epoch": 0,
        "stopped_early": False, "stop_reason": None,
    }

    best_val_mrr = 0.0
    epochs_no_improve = 0
    best_state = None
    start = time.time()

    iterator = (tqdm(range(1, epochs + 1), desc="Training",
                     mininterval=10, maxinterval=60)
                if verbose else range(1, epochs + 1))

    for epoch in iterator:
        t0 = time.time()
        show_batch = verbose and epoch <= 2
        loss = train_epoch_lcilp(
            classifier, pos_cache, optimizer,
            batch_size, device,
            x_full=data.x.to(device) if data.x is not None else None,
            drnl_max_dist=drnl_max_dist, k_neg=k_neg,
            grad_clip=grad_clip, edges_per_epoch=edges_per_epoch,
            verbose=show_batch,
        )
        epoch_time = time.time() - t0
        history["train_loss"].append(loss)
        history["epoch_times"].append(epoch_time)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        if epoch % eval_steps == 0 or epoch == epochs:
            me = None if epoch == epochs else max_eval_edges
            val_results = evaluate_ppr_lcilp(
                classifier, data, split_edge, ppr_extractor,
                split="valid", batch_size=batch_size, device=device,
                max_edges=me, cache_dir=cache_dir,
                num_negs_per_pos=eval_num_negs,
                drnl_max_dist=drnl_max_dist,
            )
            history["val_results"].append(val_results)
            mrr = val_results["mrr"]

            scheduler.step(mrr)

            if mrr > best_val_mrr + min_delta:
                best_val_mrr = mrr
                history["best_val_mrr"] = best_val_mrr
                history["best_epoch"] = epoch
                epochs_no_improve = 0
                best_state = copy.deepcopy(classifier.state_dict())
            else:
                epochs_no_improve += eval_steps

            if verbose:
                iterator.set_postfix({
                    "loss": f"{loss:.4f}",
                    "val_mrr": f"{mrr:.4f}",
                    "best": f"{best_val_mrr:.4f}",
                    "pat": f"{epochs_no_improve}/{patience}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "t/ep": f"{epoch_time:.1f}s",
                })

            if epochs_no_improve >= patience:
                history["stopped_early"] = True
                history["stop_reason"] = f"No improvement for {patience} epochs"
                if verbose:
                    print(f"\n[Early Stop] {history['stop_reason']}")
                    print(f"Best MRR: {best_val_mrr:.4f} at epoch {history['best_epoch']}")
                break

    history["total_time"] = time.time() - start

    if best_state is not None:
        classifier.load_state_dict(best_state)
        if verbose:
            print(f"Restored best model from epoch {history['best_epoch']}")

    return history
