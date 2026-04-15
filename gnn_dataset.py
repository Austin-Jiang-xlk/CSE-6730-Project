from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pseudo_labeler import PseudoLabeler, FeatureExtractor


@dataclass
class GraphSample:
    """
    One training sample centered on a target pedestrian in one time window.
    """
    x_seq: torch.Tensor           # [T, F_node_seq]
    graph_node_feats: torch.Tensor  # [N, F_graph_node]
    edge_index: torch.Tensor      # [2, E]
    edge_attr: torch.Tensor       # [E, F_edge]
    target_index: torch.Tensor    # scalar, index of target ped in graph
    y: torch.Tensor               # scalar label
    meta: Dict


class PedestrianBehaviorDataset(Dataset):
    """
    Dataset for behavior-mode classification.

    Each sample is a sliding-window target-pedestrian example:
      - x_seq: target pedestrian time-series features
      - graph snapshot: interaction graph at middle frame
      - y: pseudo label in {0,1,2,3}

    Expected label source:
      PseudoLabeler.label_scenario(...)
    Expected graph/node feature source:
      FeatureExtractor
    """

    def __init__(
        self,
        ped_csv_path: str,
        veh_csv_path: Optional[str] = None,
        labeler: Optional[PseudoLabeler] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        min_seq_len: int = 5,
        include_vehicles_in_graph: bool = True,
        cache_labels: bool = True,
    ):
        self.ped_csv_path = ped_csv_path
        self.veh_csv_path = veh_csv_path
        self.labeler = labeler or PseudoLabeler()
        self.feature_extractor = feature_extractor or FeatureExtractor(
            fps=self.labeler.fps,
            obs_len=self.labeler.window_frames,
        )
        self.min_seq_len = min_seq_len
        self.include_vehicles_in_graph = include_vehicles_in_graph
        self.cache_labels = cache_labels

        self.ped_data = self._load_ped_data(ped_csv_path)
        self.veh_data = self._load_veh_data(veh_csv_path)
        self.labels_df = self._build_or_load_labels()

        self.samples = self._build_index()

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_ped_data(self, path: str) -> pd.DataFrame:
        ped_data = pd.read_csv(path)

        if "x_est" in ped_data.columns:
            ped_data = ped_data.rename(columns={
                "x_est": "x",
                "y_est": "y",
                "vx_est": "vx",
                "vy_est": "vy",
            })

        required = {"id", "frame", "x", "y", "vx", "vy"}
        missing = required - set(ped_data.columns)
        if missing:
            raise ValueError(f"Pedestrian CSV missing columns: {sorted(missing)}")

        ped_data = ped_data.copy()
        ped_data["id"] = ped_data["id"].astype(int)
        ped_data["frame"] = ped_data["frame"].astype(int)
        return ped_data

    def _load_veh_data(self, path: Optional[str]) -> pd.DataFrame:
        if path is None or not os.path.exists(path):
            return pd.DataFrame()

        veh_data = pd.read_csv(path)

        if "x_est" in veh_data.columns:
            veh_data = veh_data.rename(columns={
                "x_est": "x",
                "y_est": "y",
                "psi_est": "psi",
                "vel_est": "vel",
            })

        if "veh_vx" not in veh_data.columns:
            if "psi" in veh_data.columns and "vel" in veh_data.columns:
                veh_data["veh_vx"] = veh_data["vel"] * np.cos(veh_data["psi"])
                veh_data["veh_vy"] = veh_data["vel"] * np.sin(veh_data["psi"])
            else:
                veh_data["veh_vx"] = 0.0
                veh_data["veh_vy"] = 0.0

        if "id" in veh_data.columns:
            veh_data["id"] = veh_data["id"].astype(int)
        if "frame" in veh_data.columns:
            veh_data["frame"] = veh_data["frame"].astype(int)

        return veh_data

    def _build_or_load_labels(self) -> pd.DataFrame:
        """
        Uses pseudo labels as supervision.
        """
        cache_path = self.ped_csv_path.replace(".csv", "_pseudo_labels.csv")

        if self.cache_labels and os.path.exists(cache_path):
            labels_df = pd.read_csv(cache_path)
        else:
            labels_df = self.labeler.label_scenario(self.ped_csv_path, self.veh_csv_path)
            if self.cache_labels:
                labels_df.to_csv(cache_path, index=False)

        required = {"ped_id", "frame_start", "frame_end", "mode", "mode_name"}
        missing = required - set(labels_df.columns)
        if missing:
            raise ValueError(f"Label DataFrame missing columns: {sorted(missing)}")

        labels_df = labels_df.copy()
        labels_df["ped_id"] = labels_df["ped_id"].astype(int)
        labels_df["frame_start"] = labels_df["frame_start"].astype(int)
        labels_df["frame_end"] = labels_df["frame_end"].astype(int)
        labels_df["mode"] = labels_df["mode"].astype(int)

        return labels_df

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------
    def _build_index(self) -> List[Dict]:
        """
        Creates a list of valid sample descriptors.
        """
        samples: List[Dict] = []

        for _, row in self.labels_df.iterrows():
            ped_id = int(row["ped_id"])
            frame_start = int(row["frame_start"])
            frame_end = int(row["frame_end"])
            mode = int(row["mode"])

            seq = self.feature_extractor.extract_node_features(
                self.ped_data,
                ped_id=ped_id,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            if seq is None or len(seq) < self.min_seq_len:
                continue

            mid_frame = frame_start + (frame_end - frame_start) // 2
            veh_data = self.veh_data if self.include_vehicles_in_graph else None

            graph_node_feats, edge_index, edge_attr, agent_ids, agent_types = \
                self.feature_extractor.build_graph_snapshot(
                    self.ped_data,
                    frame=mid_frame,
                    veh_data=veh_data,
                )

            if graph_node_feats is None or len(agent_ids) == 0:
                continue

            if ped_id not in agent_ids:
                continue

            target_index = agent_ids.index(ped_id)

            samples.append({
                "ped_id": ped_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "mid_frame": mid_frame,
                "label": mode,
                "label_name": row["mode_name"],
                "graph_node_feats": graph_node_feats,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "agent_ids": agent_ids,
                "agent_types": agent_types,
                "target_index": target_index,
                "x_seq": seq,
                "raw_label_row": row.to_dict(),
            })

        return samples

    # ------------------------------------------------------------------
    # Standard dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        item = self.samples[idx]

        x_seq = torch.tensor(item["x_seq"], dtype=torch.float32)  # [T, 7]
        graph_node_feats = torch.tensor(item["graph_node_feats"], dtype=torch.float32)  # [N, 4]

        edge_index = item["edge_index"]
        if edge_index is None or edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        edge_attr = item["edge_attr"]
        if edge_attr is None or edge_attr.size == 0:
            edge_attr = np.zeros((0, 3), dtype=np.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        target_index = torch.tensor(item["target_index"], dtype=torch.long)
        y = torch.tensor(item["label"], dtype=torch.long)

        meta = {
            "ped_id": item["ped_id"],
            "frame_start": item["frame_start"],
            "frame_end": item["frame_end"],
            "mid_frame": item["mid_frame"],
            "agent_ids": item["agent_ids"],
            "agent_types": item["agent_types"],
            "label_name": item["label_name"],
            "raw_label_row": item["raw_label_row"],
        }

        return GraphSample(
            x_seq=x_seq,
            graph_node_feats=graph_node_feats,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_index=target_index,
            y=y,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_class_distribution(self) -> Dict[int, int]:
        labels = [int(s["label"]) for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def summary(self) -> Dict:
        return {
            "num_samples": len(self.samples),
            "class_distribution": self.get_class_distribution(),
            "ped_csv_path": self.ped_csv_path,
            "veh_csv_path": self.veh_csv_path,
        }


class MultiScenarioBehaviorDataset(Dataset):
    """
    Concatenates multiple scenarios into one dataset.
    scenario_pairs:
        [
            {"ped_csv": "...", "veh_csv": "..."},
            {"ped_csv": "...", "veh_csv": None},
            ...
        ]
    """

    def __init__(
        self,
        scenario_pairs: List[Dict[str, Optional[str]]],
        labeler: Optional[PseudoLabeler] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        min_seq_len: int = 5,
        include_vehicles_in_graph: bool = True,
        cache_labels: bool = True,
    ):
        self.datasets: List[PedestrianBehaviorDataset] = []
        self.index_map: List[Tuple[int, int]] = []

        shared_labeler = labeler or PseudoLabeler()
        shared_extractor = feature_extractor or FeatureExtractor(
            fps=shared_labeler.fps,
            obs_len=shared_labeler.window_frames,
        )

        for d_idx, pair in enumerate(scenario_pairs):
            ds = PedestrianBehaviorDataset(
                ped_csv_path=pair["ped_csv"],
                veh_csv_path=pair.get("veh_csv"),
                labeler=shared_labeler,
                feature_extractor=shared_extractor,
                min_seq_len=min_seq_len,
                include_vehicles_in_graph=include_vehicles_in_graph,
                cache_labels=cache_labels,
            )
            self.datasets.append(ds)
            for local_idx in range(len(ds)):
                self.index_map.append((d_idx, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> GraphSample:
        d_idx, local_idx = self.index_map[idx]
        return self.datasets[d_idx][local_idx]

    def summary(self) -> Dict:
        class_dist: Dict[int, int] = {}
        total = 0
        for ds in self.datasets:
            total += len(ds)
            dist = ds.get_class_distribution()
            for k, v in dist.items():
                class_dist[k] = class_dist.get(k, 0) + v

        return {
            "num_scenarios": len(self.datasets),
            "num_samples": total,
            "class_distribution": class_dist,
        }


def behavior_collate_fn(batch: List[GraphSample]) -> Dict[str, object]:
    """
    Custom collate for variable-size graphs and variable-length sequences.

    Output:
        {
            "x_seq": [B tensors of shape (T_i, 7)],
            "graph_node_feats": [B tensors of shape (N_i, 4)],
            "edge_index": [B tensors of shape (2, E_i)],
            "edge_attr": [B tensors of shape (E_i, 3)],
            "target_index": tensor [B],
            "y": tensor [B],
            "meta": list[dict]
        }
    """
    return {
        "x_seq": [b.x_seq for b in batch],
        "graph_node_feats": [b.graph_node_feats for b in batch],
        "edge_index": [b.edge_index for b in batch],
        "edge_attr": [b.edge_attr for b in batch],
        "target_index": torch.stack([b.target_index for b in batch], dim=0),
        "y": torch.stack([b.y for b in batch], dim=0),
        "meta": [b.meta for b in batch],
    }


def discover_scenario_pairs(data_dir: str) -> List[Dict[str, Optional[str]]]:
    """
    Recursively discover scenario pairs under data_dir.

    Rules:
    - ped files contain "_traj_ped_"
    - veh files contain "_traj_veh_"
    - skip generated pseudo-label caches
    - use (relative folder + scenario prefix) as grouping key
    """
    scenario_dict: Dict[str, Dict[str, str]] = {}

    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue
            if f.endswith("_pseudo_labels.csv"):
                continue

            full_path = os.path.join(root, f)

            if "_traj_" not in f:
                continue

            prefix = f.split("_traj_")[0]
            rel_root = os.path.relpath(root, data_dir)
            key = os.path.join(rel_root, prefix)

            scenario_dict.setdefault(key, {})

            if "_traj_ped_" in f:
                scenario_dict[key]["ped_csv"] = full_path
            elif "_traj_veh_" in f:
                scenario_dict[key]["veh_csv"] = full_path

    pairs = []
    for _, info in scenario_dict.items():
        if "ped_csv" not in info:
            continue
        pairs.append({
            "ped_csv": info["ped_csv"],
            "veh_csv": info.get("veh_csv"),
        })

    return sorted(pairs, key=lambda x: x["ped_csv"])