# gnn_model.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleGraphConv(nn.Module):
    """
    A lightweight graph layer without external libraries.

    Message:
        m_ij = MLP([x_i, x_j, e_ij])

    Aggregate:
        mean over incoming neighbors

    Update:
        x_i' = MLP([x_i, agg_i])
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.message_mlp = MLP(
            in_dim=node_dim * 2 + edge_dim,
            hidden_dims=[hidden_dim],
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.update_mlp = MLP(
            in_dim=node_dim + hidden_dim,
            hidden_dims=[hidden_dim],
            out_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # [N, Dn]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E, De]
    ) -> torch.Tensor:
        device = node_feats.device
        N = node_feats.size(0)

        if edge_index.numel() == 0:
            zero_agg = torch.zeros(N, self.update_mlp.net[0].in_features - node_feats.size(1), device=device)
            return self.update_mlp(torch.cat([node_feats, zero_agg], dim=-1))

        src = edge_index[0]  # source node indices
        dst = edge_index[1]  # target node indices

        x_src = node_feats[src]  # [E, Dn]
        x_dst = node_feats[dst]  # [E, Dn]

        msg_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)  # [E, 2Dn+De]
        messages = self.message_mlp(msg_input)  # [E, H]

        hidden_dim = messages.size(-1)
        agg = torch.zeros(N, hidden_dim, device=device)
        deg = torch.zeros(N, 1, device=device)

        agg.index_add_(0, dst, messages)
        deg.index_add_(0, dst, torch.ones(messages.size(0), 1, device=device))

        agg = agg / deg.clamp(min=1.0)

        out = self.update_mlp(torch.cat([node_feats, agg], dim=-1))
        return out


class TemporalEncoder(nn.Module):
    """
    Encodes the target pedestrian trajectory sequence.
    Input shape: [T, F]
    Output shape: [H]
    """
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [T, F] or [1, T, F]
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # [1, T, F]

        _, h_n = self.gru(x_seq)
        if self.bidirectional:
            out = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [1, 2H]
        else:
            out = h_n[-1]  # [1, H]
        return out.squeeze(0)  # [H] or [2H]


class GraphEncoder(nn.Module):
    """
    Encodes one graph snapshot and returns node embeddings.
    """
    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SimpleGraphConv(
                node_dim=hidden_dim,
                edge_dim=edge_input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def forward(
        self,
        graph_node_feats: torch.Tensor,  # [N, 4]
        edge_index: torch.Tensor,        # [2, E]
        edge_attr: torch.Tensor,         # [E, 3]
    ) -> torch.Tensor:
        x = self.node_proj(graph_node_feats)  # [N, H]

        for conv, norm in zip(self.layers, self.norms):
            h = conv(x, edge_index, edge_attr)
            x = norm(x + F.dropout(h, p=self.dropout, training=self.training))
            x = F.relu(x)

        return x  # [N, H]


class BehaviorGNN(nn.Module):
    """
    Hybrid behavior classifier:
      - Temporal encoder over target pedestrian sequence
      - Graph encoder over snapshot interaction graph
      - Classification head over concatenated embeddings

    Output classes:
      0 aggressive
      1 regular
      2 cautious
      3 following
    """
    def __init__(
        self,
        seq_input_dim: int = 7,
        graph_node_dim: int = 4,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        temporal_layers: int = 1,
        graph_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
        bidirectional_temporal: bool = False,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            input_dim=seq_input_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=bidirectional_temporal,
        )

        self.graph_encoder = GraphEncoder(
            node_input_dim=graph_node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            dropout=dropout,
        )

        fusion_dim = self.temporal_encoder.output_dim + hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_single(
        self,
        x_seq: torch.Tensor,            # [T, 7]
        graph_node_feats: torch.Tensor, # [N, 4]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, 3]
        target_index: torch.Tensor,     # scalar
    ) -> torch.Tensor:
        seq_emb = self.temporal_encoder(x_seq)  # [H_seq]

        node_embs = self.graph_encoder(graph_node_feats, edge_index, edge_attr)  # [N, H]
        graph_target_emb = node_embs[target_index]  # [H]

        fused = torch.cat([seq_emb, graph_target_emb], dim=-1)  # [H_seq + H]
        logits = self.classifier(fused)  # [4]
        return logits

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Expects output from behavior_collate_fn in gnn_dataset.py
        Returns:
            logits: [B, 4]
        """
        logits_list = []
        batch_size = len(batch["x_seq"])

        for i in range(batch_size):
            logits_i = self.forward_single(
                x_seq=batch["x_seq"][i],
                graph_node_feats=batch["graph_node_feats"][i],
                edge_index=batch["edge_index"][i],
                edge_attr=batch["edge_attr"][i],
                target_index=batch["target_index"][i],
            )
            logits_list.append(logits_i)

        return torch.stack(logits_list, dim=0)  # [B, 4]

    @torch.no_grad()
    def predict_proba(self, batch: dict) -> torch.Tensor:
        logits = self.forward(batch)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, batch: dict) -> torch.Tensor:
        return self.predict_proba(batch).argmax(dim=-1)


def build_behavior_gnn(
    hidden_dim: int = 64,
    temporal_layers: int = 1,
    graph_layers: int = 2,
    dropout: float = 0.1,
) -> BehaviorGNN:
    """
    Convenience factory.
    """
    return BehaviorGNN(
        seq_input_dim=7,
        graph_node_dim=4,
        edge_dim=3,
        hidden_dim=hidden_dim,
        temporal_layers=temporal_layers,
        graph_layers=graph_layers,
        num_classes=4,
        dropout=dropout,
        bidirectional_temporal=False,
    )