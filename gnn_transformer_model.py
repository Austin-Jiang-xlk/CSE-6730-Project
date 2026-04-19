from __future__ import annotations
from typing import List
import math
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
    Message: m_ij = MLP([x_i, x_j, e_ij])
    Aggregate: mean over incoming neighbors
    Update: x_i' = MLP([x_i, agg_i])
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
            hidden_dim = self.update_mlp.net[-1].out_features
            zero_agg = torch.zeros(N, hidden_dim, device=device)
            return self.update_mlp(torch.cat([node_feats, zero_agg], dim=-1))

        src = edge_index[0]
        dst = edge_index[1]

        x_src = node_feats[src]
        x_dst = node_feats[dst]

        msg_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)  # [E, H]

        hidden_dim = messages.size(-1)
        agg = torch.zeros(N, hidden_dim, device=device)
        deg = torch.zeros(N, 1, device=device)

        agg.index_add_(0, dst, messages)
        deg.index_add_(0, dst, torch.ones(messages.size(0), 1, device=device))
        agg = agg / deg.clamp(min=1.0)

        out = self.update_mlp(torch.cat([node_feats, agg], dim=-1))
        return out


class GraphEncoder(nn.Module):
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
        x = self.node_proj(graph_node_feats)
        for conv, norm in zip(self.layers, self.norms):
            h = conv(x, edge_index, edge_attr)
            x = norm(x + F.dropout(h, p=self.dropout, training=self.training))
            x = F.relu(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerTemporalEncoder(nn.Module):
    """
    Replaces the GRU-based TemporalEncoder.
    Input:  [T, F] or [B, T, F]
    Output: [D] or [B, D]
    """
    def __init__(
        self,
        input_dim: int = 7,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 256,
        pooling: str = "cls",  # "cls" or "mean"
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        else:
            self.cls_token = None

        self.norm = nn.LayerNorm(model_dim)

    @property
    def output_dim(self) -> int:
        return self.model_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [T, F] or [B, T, F]
        """
        single_input = False
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # [1, T, F]
            single_input = True

        x = self.input_proj(x_seq)  # [B, T, D]

        if self.pooling == "cls":
            B = x.size(0)
            cls = self.cls_token.expand(B, 1, self.model_dim)
            x = torch.cat([cls, x], dim=1)  # [B, T+1, D]

        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.pooling == "cls":
            out = x[:, 0, :]   # [B, D]
        else:
            out = x.mean(dim=1)  # [B, D]

        if single_input:
            out = out.squeeze(0)

        return out


class BehaviorGNNTransformer(nn.Module):
    """
    Hybrid classifier:
    - Transformer encoder over target pedestrian sequence
    - Graph encoder over interaction graph snapshot
    - Fusion head for 4 behavior classes
    """
    def __init__(
        self,
        seq_input_dim: int = 7,
        graph_node_dim: int = 4,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 128,
        graph_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()

        self.temporal_encoder = TransformerTemporalEncoder(
            input_dim=seq_input_dim,
            model_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            pooling=pooling,
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
        x_seq: torch.Tensor,           # [T, 7]
        graph_node_feats: torch.Tensor, # [N, 4]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, 3]
        target_index: torch.Tensor,     # scalar
    ) -> torch.Tensor:
        seq_emb = self.temporal_encoder(x_seq)  # [H]
        node_embs = self.graph_encoder(graph_node_feats, edge_index, edge_attr)  # [N, H]
        graph_target_emb = node_embs[target_index]  # [H]

        fused = torch.cat([seq_emb, graph_target_emb], dim=-1)
        logits = self.classifier(fused)
        return logits

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Same batch structure as your current BehaviorGNN.
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


def build_behavior_gnn_transformer(
    hidden_dim: int = 64,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    transformer_ff_dim: int = 128,
    graph_layers: int = 2,
    dropout: float = 0.1,
    pooling: str = "cls",
) -> BehaviorGNNTransformer:
    return BehaviorGNNTransformer(
        seq_input_dim=7,
        graph_node_dim=4,
        edge_dim=3,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
        graph_layers=graph_layers,
        num_classes=4,
        dropout=dropout,
        pooling=pooling,
    )