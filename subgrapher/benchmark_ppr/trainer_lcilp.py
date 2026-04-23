"""Training pipeline for Static PPR subgraph link prediction.

Architecture: SubgraphClassifier (DRNL + content features → GNN → graph-level MLP).
Loss: 2-way cross-entropy with one rotating global negative per positive per epoch.

Negative sampling strategy:
  K pre-cached negative subgraph caches are built (each uses a different random seed).
  Each training epoch uses a different cache, cycling: epoch i uses neg_caches[i % K].
  A positive (u,v) sees neg_cache_0 on epoch 0, neg_cache_1 on epoch 1, ..., then repeats.

  Why rotate rather than use all K simultaneously:
    Using all K at once turns training into a K+1-way CE task. If the model satisfies
    this in a few epochs (trivially separating G_{u,v} from the K fixed G_{u,n_k}),
    gradients collapse and training stops. Rotating ensures the model always encounters
    at least one "unsatisfied" negative each epoch, keeping gradients alive.
    After N epochs the model has seen K×(N/K) = N different negatives per positive,
    which builds genuine generalisation to the 100-neg eval task.

  Why global (not in-subgraph) negatives with DRNL:
    DRNL encodes d(node, query_v) for every node in the subgraph. In-subgraph negatives
    (other nodes w inside G_{u,v}) all have d(w,v)>=1, while v itself has d(v,v)=0.
    SAGE trivially exploits this label to identify v with no structural learning.
    Global negatives compare G_{u,v} vs G_{u,n} as separate subgraphs — in G_{u,n},
    n is the query node with d(n,n)=0, so DRNL is equally informative for both and
    the model must learn whether the overall subgraph topology looks like a link exists.
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
from .evaluator import evaluate_ppr_lcilp


# ---------------------------------------------------------------------------
# Negative cache construction
# ---------------------------------------------------------------------------

def build_or_load_neg_csr_cache(source_edge, target_edge, data, ppr_extractor,
                                 num_nodes, cache_dir=None, seed=42,
                                 verbose=True):
    """Build (or load) one global-negative subgraph per training edge."""
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


def build_or_load_neg_csr_caches(source_edge, target_edge, data, ppr_extractor,
                                   num_nodes, cache_dir=None, num_negs=5,
                                   verbose=True):
    """Build/load K independent neg CSR caches (one per random seed).

    The legacy single-neg cache (train_neg_subgraphs_csr.pt, seed=42) is
    reused as k=0 when it exists.
    """
    caches = []
    for k in range(num_negs):
        seed = 42 + k * 17
        path = (os.path.join(cache_dir, f"train_neg_subgraphs_csr_{k}.pt")
                if cache_dir else None)
        old_path = (os.path.join(cache_dir, "train_neg_subgraphs_csr.pt")
                    if cache_dir else None)

        if path and os.path.isfile(path):
            if verbose:
                print(f"[Cache] Loading neg CSR {k}/{num_negs} from {path}")
            cache = SubgraphCSR.load(path)
        elif k == 0 and old_path and os.path.isfile(old_path):
            if verbose:
                print(f"[Cache] Reusing legacy neg CSR as k=0 ({old_path})")
            cache = SubgraphCSR.load(old_path)
            if path:
                cache.save(path)
        else:
            if verbose:
                print(f"[Cache] Building neg CSR {k}/{num_negs} (seed={seed})...")
            cache = build_or_load_neg_csr_cache(
                source_edge, target_edge, data, ppr_extractor,
                num_nodes=num_nodes, cache_dir=None, seed=seed, verbose=verbose)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cache.save(path)
                if verbose:
                    mb = os.path.getsize(path) / 1e6
                    print(f"[Cache] Saved neg CSR {k}: {path} ({mb:.0f} MB)")

        caches.append(cache)
    return caches


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch_lcilp(classifier, pos_cache, neg_cache, optimizer,
                      batch_size, device, x_full=None,
                      grad_clip=1.0, edges_per_epoch=None, verbose=False):
    """One epoch: 2-way CE with a single global neg cache.

    neg_cache is kept on CPU; pos_cache is on device.
    PPR structural features must be pre-computed in cache.drnl_feats before calling.
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

    x_full_cpu = x_full.cpu()

    total_loss, total_examples = 0.0, 0

    for perm in dataloader:
        idx = torch.as_tensor(perm, dtype=torch.long, device=device)

        both_mask = pos_cache.valid_mask[idx] & neg_cache.valid_mask[idx.cpu()].to(device)
        both_idx = idx[both_mask]
        if both_idx.numel() == 0:
            continue

        pos_b = pos_cache.make_batch(both_idx, x_full)
        B = int(pos_b["u_idx"].size(0))
        if B == 0 or pos_b["total_edges"] == 0:
            continue

        pos_x = torch.cat([pos_b["drnl_x"].to(device),
                            pos_b["x"].to(device)], dim=-1)

        pos_bvec = torch.repeat_interleave(
            torch.arange(B, device=device), pos_b["num_nodes_vec"])

        neg_b = neg_cache.make_batch(both_idx.cpu(), x_full_cpu)
        n_B = int(neg_b["u_idx"].size(0))
        if n_B == 0:
            continue

        neg_x = torch.cat([neg_b["drnl_x"].to(device),
                            neg_b["x"].to(device)], dim=-1)

        neg_bvec = torch.repeat_interleave(
            torch.arange(n_B, device=device),
            neg_b["num_nodes_vec"].to(device))

        optimizer.zero_grad()
        score_pos = classifier(pos_x, pos_b["edge_index"], pos_bvec,
                               pos_b["u_idx"], pos_b["v_idx"])
        score_neg = classifier(neg_x, neg_b["edge_index"].to(device), neg_bvec,
                               neg_b["u_idx"].to(device), neg_b["v_idx"].to(device))

        min_B = min(score_pos.size(0), score_neg.size(0))
        logits = torch.stack([score_pos[:min_B], score_neg[:min_B]], dim=1)
        target = torch.zeros(min_B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target)
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
                           ppr_preprocessor,
                           pos_cache=None,
                           epochs=150, batch_size=256, lr=0.01,
                           eval_steps=5, device="cpu", verbose=True,
                           patience=10, min_delta=0.0001,
                           weight_decay=5e-4, grad_clip=1.0,
                           num_negs=5, edges_per_epoch=None,
                           cache_dir=None, max_eval_edges=2000,
                           eval_num_negs=None, **_kwargs):
    """Train SubgraphClassifier with rotating global negatives and PPR-score labels.

    Parameters
    ----------
    ppr_preprocessor : PPRPreprocessor
        Pre-computed per-node PPR vectors.  Used to build (π_u, π_v) structural
        features for every node in each subgraph (replaces DRNL).
    num_negs : int
        Number of independent negative caches (K).  Epoch i uses
        neg_caches[i % K], cycling through all K caches.
    """
    from .trainer_batched import build_or_load_ppr_csr_cache
    from ..utils.drnl import compute_ppr_feats_for_csr

    classifier = classifier.to(device)

    source_edge = split_edge["train"]["source_node"]
    target_edge = split_edge["train"]["target_node"]

    # --- positive cache ---
    if pos_cache is None:
        pos_cache = build_or_load_ppr_csr_cache(
            source_edge, target_edge, data, ppr_extractor,
            cache_dir=cache_dir, verbose=verbose,
        )

    # --- K negative caches (CPU) ---
    neg_caches = build_or_load_neg_csr_caches(
        source_edge, target_edge, data, ppr_extractor,
        num_nodes=data.num_nodes,
        cache_dir=cache_dir, num_negs=num_negs, verbose=verbose,
    )

    # --- pre-compute PPR features once for pos + all neg caches ---
    ppr_items = [("pos", pos_cache, "train_subgraphs_csr.pt")] + [
        (f"neg_{k}", neg_caches[k], f"train_neg_subgraphs_csr_{k}.pt")
        for k in range(len(neg_caches))
    ]
    for tag, cache, fname in ppr_items:
        if cache.drnl_feats is None or cache.drnl_feats.shape[1] != 2:
            if verbose:
                print(f"[PPR] Pre-computing PPR features for {tag} cache "
                      f"({int(cache.valid_mask.sum())} subgraphs)...")
            cache.drnl_feats = compute_ppr_feats_for_csr(
                cache, ppr_preprocessor.ppr_cache, verbose=verbose)
            if cache_dir:
                path = os.path.join(cache_dir, fname)
                cache.save(path)
                if verbose:
                    mb = os.path.getsize(path) / 1e6
                    print(f"[PPR] Saved {tag} cache with PPR feats: {path} ({mb:.0f} MB)")

    # pos_cache on device; neg_caches stay on CPU
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
        # Rotate: each epoch uses a different neg cache
        neg_cache = neg_caches[(epoch - 1) % len(neg_caches)]

        t0 = time.time()
        show_batch = verbose and epoch <= 2
        loss = train_epoch_lcilp(
            classifier, pos_cache, neg_cache, optimizer,
            batch_size, device,
            x_full=data.x.to(device) if data.x is not None else None,
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
                ppr_preprocessor=ppr_preprocessor,
                split="valid", batch_size=batch_size, device=device,
                max_edges=me, cache_dir=cache_dir,
                num_negs_per_pos=eval_num_negs,
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
                    "neg": f"k{(epoch-1) % len(neg_caches)}",
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
