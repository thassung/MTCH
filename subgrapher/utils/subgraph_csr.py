"""
CSR-style in-memory cache of per-edge subgraphs.

Used by the k-hop, static-PPR and learnable-PPR training/eval pipelines to
replace the old pickled list-of-tensors cache (which pickled 50 GB for k=1
FB15K237 and thrashed disk on every batch).

Layout for N edges with per-edge subgraphs:

  node_ids    LongTensor [total_nodes]       concat of each subgraph's global node IDs
  node_offs   LongTensor [N + 1]             CSR offsets into node_ids
  edge_src    LongTensor [total_edges]       LOCAL src indices (already relabeled)
  edge_dst    LongTensor [total_edges]       LOCAL dst indices (already relabeled)
  edge_offs   LongTensor [N + 1]             CSR offsets into edge_src / edge_dst
  u_sub       LongTensor [N]                 local position of positive source in its subgraph
  v_sub       LongTensor [N]                 local position of positive target in its subgraph
  num_nodes   LongTensor [N]                 sizes (= node_offs[i+1] - node_offs[i])
  valid_mask  BoolTensor [N]                 False for edges whose (u, v) fell outside any subgraph

Empty subgraphs (u or v missing) are stored as zero-length slices with
u_sub = v_sub = 0 and valid_mask[i] = False. Callers should skip i where
valid_mask[i] is False.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class SubgraphCSR:
    node_ids: torch.Tensor
    node_offs: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_offs: torch.Tensor
    u_sub: torch.Tensor
    v_sub: torch.Tensor
    num_nodes: torch.Tensor
    valid_mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.u_sub.size(0))

    def to(self, device) -> "SubgraphCSR":
        return SubgraphCSR(
            node_ids=self.node_ids.to(device),
            node_offs=self.node_offs.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            edge_offs=self.edge_offs.to(device),
            u_sub=self.u_sub.to(device),
            v_sub=self.v_sub.to(device),
            num_nodes=self.num_nodes.to(device),
            valid_mask=self.valid_mask.to(device),
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        num_edges: int,
        extract_fn: Callable[[int], tuple],
        progress_desc: str = "Building CSR cache",
        verbose: bool = True,
    ) -> "SubgraphCSR":
        """Build a CSR cache by calling extract_fn(i) for each i in range(num_edges).

        extract_fn(i) must return either
            (selected_nodes_long_1d, edge_index_long_2xE, u_sub_int, v_sub_int)
        or None for an invalid edge (stored as an empty subgraph, valid=False).

        Indices in edge_index_long_2xE must already be locally relabeled to
        [0, len(selected_nodes)).
        """
        from tqdm import tqdm

        node_chunks = []
        edge_src_chunks = []
        edge_dst_chunks = []
        node_offs = torch.zeros(num_edges + 1, dtype=torch.long)
        edge_offs = torch.zeros(num_edges + 1, dtype=torch.long)
        u_sub = torch.zeros(num_edges, dtype=torch.long)
        v_sub = torch.zeros(num_edges, dtype=torch.long)
        num_nodes = torch.zeros(num_edges, dtype=torch.long)
        valid_mask = torch.zeros(num_edges, dtype=torch.bool)

        it = tqdm(range(num_edges), desc=progress_desc, leave=False,
                  mininterval=10) if verbose else range(num_edges)

        n_running = 0
        e_running = 0
        skipped = 0
        for i in it:
            result = extract_fn(i)
            if result is None:
                node_offs[i + 1] = n_running
                edge_offs[i + 1] = e_running
                skipped += 1
                continue

            sel, ei, u_s, v_s = result
            sel = sel.to(torch.long).cpu()
            ei = ei.to(torch.long).cpu()
            nn_i = int(sel.size(0))
            ne_i = int(ei.size(1))

            node_chunks.append(sel)
            if ne_i > 0:
                edge_src_chunks.append(ei[0])
                edge_dst_chunks.append(ei[1])

            n_running += nn_i
            e_running += ne_i
            node_offs[i + 1] = n_running
            edge_offs[i + 1] = e_running
            u_sub[i] = int(u_s)
            v_sub[i] = int(v_s)
            num_nodes[i] = nn_i
            valid_mask[i] = True

        if verbose and skipped:
            print(f"  ({skipped}/{num_edges} edges had no valid subgraph)")

        node_ids = torch.cat(node_chunks) if node_chunks else torch.zeros(0, dtype=torch.long)
        edge_src = torch.cat(edge_src_chunks) if edge_src_chunks else torch.zeros(0, dtype=torch.long)
        edge_dst = torch.cat(edge_dst_chunks) if edge_dst_chunks else torch.zeros(0, dtype=torch.long)

        return cls(
            node_ids=node_ids,
            node_offs=node_offs,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_offs=edge_offs,
            u_sub=u_sub,
            v_sub=v_sub,
            num_nodes=num_nodes,
            valid_mask=valid_mask,
        )

    # ------------------------------------------------------------------
    # Disk persistence (flat tensors only, no pickled Python lists)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "format": "SubgraphCSR/v1",
                "node_ids": self.node_ids,
                "node_offs": self.node_offs,
                "edge_src": self.edge_src,
                "edge_dst": self.edge_dst,
                "edge_offs": self.edge_offs,
                "u_sub": self.u_sub,
                "v_sub": self.v_sub,
                "num_nodes": self.num_nodes,
                "valid_mask": self.valid_mask,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SubgraphCSR":
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d.get("format") != "SubgraphCSR/v1":
            raise ValueError(f"{path} is not a SubgraphCSR/v1 file")
        return cls(
            node_ids=d["node_ids"],
            node_offs=d["node_offs"],
            edge_src=d["edge_src"],
            edge_dst=d["edge_dst"],
            edge_offs=d["edge_offs"],
            u_sub=d["u_sub"],
            v_sub=d["v_sub"],
            num_nodes=d["num_nodes"],
            valid_mask=d["valid_mask"],
        )

    # ------------------------------------------------------------------
    # Vectorised batch construction
    # ------------------------------------------------------------------

    def make_batch(self, idx: torch.Tensor, x_full: torch.Tensor) -> dict:
        """Build a GNN-ready batch for the edge indices in idx.

        idx is a LongTensor on the same device as the CSR tensors (call .to()
        first). Invalid edges in idx are silently dropped.
        """
        device = self.node_ids.device

        keep_mask = self.valid_mask[idx]
        valid_idx = idx[keep_mask]
        B = int(valid_idx.size(0))
        if B == 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            return {
                "x": x_full[empty_long],
                "edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
                "u_idx": empty_long,
                "v_idx": empty_long,
                "num_nodes_vec": empty_long,
                "batch_node_offsets": torch.zeros(1, dtype=torch.long, device=device),
                "valid_idx": valid_idx,
                "total_nodes": 0,
                "total_edges": 0,
            }

        node_lo = self.node_offs[valid_idx]
        node_hi = self.node_offs[valid_idx + 1]
        edge_lo = self.edge_offs[valid_idx]
        edge_hi = self.edge_offs[valid_idx + 1]
        nn_b = node_hi - node_lo
        ne_b = edge_hi - edge_lo

        total_nodes = int(nn_b.sum().item())
        total_edges = int(ne_b.sum().item())

        batch_node_offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        batch_node_offsets[1:] = torch.cumsum(nn_b, dim=0)

        node_batch_vec = torch.repeat_interleave(
            torch.arange(B, device=device), nn_b
        )
        node_local_pos = (
            torch.arange(total_nodes, device=device)
            - batch_node_offsets[node_batch_vec]
        )
        node_gather = node_lo[node_batch_vec] + node_local_pos
        sel_concat = self.node_ids[node_gather]

        x_batch = x_full[sel_concat]

        if total_edges > 0:
            batch_edge_offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
            batch_edge_offsets[1:] = torch.cumsum(ne_b, dim=0)

            edge_batch_vec = torch.repeat_interleave(
                torch.arange(B, device=device), ne_b
            )
            edge_local_pos = (
                torch.arange(total_edges, device=device)
                - batch_edge_offsets[edge_batch_vec]
            )
            edge_gather = edge_lo[edge_batch_vec] + edge_local_pos

            es = self.edge_src[edge_gather] + batch_node_offsets[edge_batch_vec]
            ed = self.edge_dst[edge_gather] + batch_node_offsets[edge_batch_vec]
            edge_index = torch.stack([es, ed], dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        u_idx_batch = batch_node_offsets[:B] + self.u_sub[valid_idx]
        v_idx_batch = batch_node_offsets[:B] + self.v_sub[valid_idx]

        return {
            "x": x_batch,
            "edge_index": edge_index,
            "u_idx": u_idx_batch,
            "v_idx": v_idx_batch,
            "num_nodes_vec": nn_b,
            "batch_node_offsets": batch_node_offsets,
            "valid_idx": valid_idx,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    def summary(self) -> dict:
        valid = int(self.valid_mask.sum().item())
        nn = self.num_nodes[self.valid_mask]
        return {
            "num_edges": int(self.valid_mask.size(0)),
            "num_valid": valid,
            "total_nodes": int(self.node_ids.size(0)),
            "total_edges": int(self.edge_src.size(0)),
            "avg_subgraph_nodes": float(nn.float().mean().item()) if valid > 0 else 0.0,
            "max_subgraph_nodes": int(nn.max().item()) if valid > 0 else 0,
        }
