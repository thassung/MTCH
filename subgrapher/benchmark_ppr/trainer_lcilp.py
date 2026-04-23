"""LCILP-faithful training pipeline for Static PPR subgraph link prediction.

Key differences from trainer_batched.py (old BCE + in-subgraph neg approach):

  1. DRNL node features  — structural distance labels, not random embeddings
  2. Global negatives    — one pre-cached neg subgraph per positive edge
                           (corrupt the tail, extract PPR({u, n_neg}))
  3. MarginRankingLoss   — margin=10, same as LCILP
  4. SubgraphClassifier  — graph-level pooling score, not node-pair dot product
  5. Subgraph-only eval  — validation and test MRR via pair-subgraph scoring
                           (no full-graph eval = no distribution shift)
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
# Negative cache construction
# ---------------------------------------------------------------------------

def build_or_load_neg_csr_cache(source_edge, target_edge, data, ppr_extractor,
                                 num_nodes, cache_dir=None, seed=42,
                                 verbose=True):
    """Build (or load) one global-negative subgraph per training edge.

    For each positive (u, v), samples a random entity n_neg ≠ v and extracts
    PPR({u, n_neg}).  The resulting SubgraphCSR is aligned with the positive
    cache: neg_cache[i] corresponds to pos_cache[i].
    """
    if cache_dir:
        path = os.path.join(cache_dir, "train_neg_subgraphs_csr.pt")
        if os.path.isfile(path):
            if verbose:
                print(f"[Cache] Loading neg CSR from {path}")
            return SubgraphCSR.load(path)

    if verbose:
        print("[Cache] Extracting negative PPR subgraphs (one-time cost)...")

    rng = torch.Generator().manual_seed(seed)

    def extract(i: int):
        u = int(source_edge[i].item())
        v = int(target_edge[i].item())

        # Sample a global negative tail ≠ v
        n_neg = int(torch.randint(num_nodes, (1,), generator=rng).item())
        for _ in range(50):
            if n_neg != v:
                break
            n_neg = int(torch.randint(num_nodes, (1,), generator=rng).item())

        sub_data, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, n_neg)
        u_sub = metadata.get("u_subgraph", -1)
        neg_sub = metadata.get("v_subgraph", -1)
        if u_sub == -1 or neg_sub == -1:
            return None
        return (
            selected_nodes.to(torch.long),
            sub_data.edge_index.to(torch.long),
            int(u_sub),
            int(neg_sub),
        )

    cache = SubgraphCSR.build(
        num_edges=int(source_edge.size(0)),
        extract_fn=extract,
        progress_desc="Building neg PPR CSR",
        verbose=verbose,
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "train_neg_subgraphs_csr.pt")
        cache.save(path)
        if verbose:
            mb = os.path.getsize(path) / 1e6
            print(f"[Cache] Saved neg CSR: {path} ({mb:.0f} MB)")
            print(f"[Cache] {cache.summary()}")

    return cache


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch_lcilp(classifier, pos_cache, neg_cache, optimizer,
                      batch_size, device, x_full=None, drnl_max_dist=6,
                      grad_clip=1.0, edges_per_epoch=None, verbose=False):
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

    # x_full used by make_batch for content features; fall back to zeros if None.
    if x_full is None:
        n_nodes = int(pos_cache.node_ids.max().item()) + 1
        x_full = torch.zeros(n_nodes, 1, device=device)

    total_loss, total_examples = 0.0, 0

    for perm in dataloader:
        idx = torch.as_tensor(perm, dtype=torch.long, device=device)

        # Only process indices valid in BOTH pos and neg caches
        both_mask = pos_cache.valid_mask[idx] & neg_cache.valid_mask[idx]
        both_idx = idx[both_mask]
        if both_idx.numel() == 0:
            continue

        pos_b = pos_cache.make_batch(both_idx, x_full)
        neg_b = neg_cache.make_batch(both_idx, x_full)
        B = int(pos_b["u_idx"].size(0))
        if B == 0 or pos_b["total_edges"] == 0:
            continue

        # Use pre-computed DRNL; concatenate with content features if present
        if pos_b["drnl_x"] is not None:
            pos_x = torch.cat([pos_b["drnl_x"].to(device),
                                pos_b["x"].to(device)], dim=-1)
            neg_x = torch.cat([neg_b["drnl_x"].to(device),
                                neg_b["x"].to(device)], dim=-1)
        else:
            pos_x = compute_drnl_for_batch(pos_b, max_dist=drnl_max_dist)
            neg_x = compute_drnl_for_batch(neg_b, max_dist=drnl_max_dist)

        # batch assignment vector for global_mean_pool
        pos_bvec = torch.repeat_interleave(
            torch.arange(B, device=device), pos_b["num_nodes_vec"])
        neg_bvec = torch.repeat_interleave(
            torch.arange(B, device=device), neg_b["num_nodes_vec"])

        optimizer.zero_grad()
        score_pos = classifier(pos_x, pos_b["edge_index"], pos_bvec,
                               pos_b["u_idx"], pos_b["v_idx"])
        score_neg = classifier(neg_x, neg_b["edge_index"], neg_bvec,
                               neg_b["u_idx"], neg_b["v_idx"])

        # Cross-entropy ranking: stack [pos, neg] → target always index 0
        min_B = min(score_pos.size(0), score_neg.size(0))
        scores_stacked = torch.stack([score_pos[:min_B], score_neg[:min_B]], dim=1)
        target = torch.zeros(min_B, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores_stacked, target)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * min_B
        total_examples += min_B

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
                           drnl_max_dist=6,
                           edges_per_epoch=None, cache_dir=None,
                           max_eval_edges=2000, eval_num_negs=None,
                           **_kwargs):
    """Train SubgraphClassifier with LCILP-faithful pipeline.

    Parameters
    ----------
    pos_cache : SubgraphCSR or None
        Pre-built positive subgraph cache.  If None, it is built from
        split_edge['train'] using ppr_extractor.
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

    # --- negative cache ---
    neg_cache = build_or_load_neg_csr_cache(
        source_edge, target_edge, data, ppr_extractor,
        num_nodes=data.num_nodes,
        cache_dir=cache_dir, seed=42, verbose=verbose,
    )

    # --- pre-compute DRNL once (avoids Python BFS every training batch) ---
    from ..utils.drnl import compute_drnl_for_csr
    for tag, cache, fname in [
        ("pos", pos_cache, "train_subgraphs_csr.pt"),
        ("neg", neg_cache, "train_neg_subgraphs_csr.pt"),
    ]:
        if cache.drnl_feats is None:
            if verbose:
                print(f"[DRNL] Pre-computing DRNL for {tag} cache "
                      f"({int(cache.valid_mask.sum())} subgraphs)...")
            cache.drnl_feats = compute_drnl_for_csr(
                cache, max_dist=drnl_max_dist, verbose=verbose)
            if cache_dir:
                path = os.path.join(cache_dir, fname)
                cache.save(path)
                if verbose:
                    mb = os.path.getsize(path) / 1e6
                    print(f"[DRNL] Saved {tag} cache with DRNL: {path} ({mb:.0f} MB)")

    pos_cache = pos_cache.to(device)
    neg_cache = neg_cache.to(device)

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
            classifier, pos_cache, neg_cache, optimizer,
            batch_size, device,
            x_full=data.x.to(device) if data.x is not None else None,
            drnl_max_dist=drnl_max_dist,
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
