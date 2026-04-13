from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_dataset import (
    PedestrianBehaviorDataset,
    MultiScenarioBehaviorDataset,
    behavior_collate_fn,
    discover_scenario_pairs,
)
from gnn_model import build_behavior_gnn


MODE_NAMES = ["aggressive", "regular", "cautious", "following"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GNN inference and export per-pedestrian mode probabilities."
    )

    # model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint .pt",
    )

    # input data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing multiple scenario CSVs",
    )
    parser.add_argument(
        "--ped_csv",
        type=str,
        default=None,
        help="Single pedestrian CSV path",
    )
    parser.add_argument(
        "--veh_csv",
        type=str,
        default=None,
        help="Single vehicle CSV path",
    )

    # dataset
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_vehicle_graph", action="store_true")
    parser.add_argument("--no_cache_labels", action="store_true")

    # aggregation
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "max_conf", "latest"],
        help=(
            "How to aggregate multiple window-level predictions for the same pedestrian. "
            "mean = average probs across windows; "
            "max_conf = keep the most confident window; "
            "latest = keep the latest window."
        ),
    )

    # output
    parser.add_argument(
        "--output_json",
        type=str,
        default="mode_probabilities.json",
        help="Where to save per-pedestrian mode probabilities",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional csv export path",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_args = ckpt.get("args", {})
    hidden_dim = saved_args.get("hidden_dim", 64)
    temporal_layers = saved_args.get("temporal_layers", 1)
    graph_layers = saved_args.get("graph_layers", 2)
    dropout = saved_args.get("dropout", 0.1)

    model = build_behavior_gnn(
        hidden_dim=hidden_dim,
        temporal_layers=temporal_layers,
        graph_layers=graph_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_dataset(args: argparse.Namespace):
    if args.data_dir:
        scenario_pairs = discover_scenario_pairs(args.data_dir)
        if len(scenario_pairs) == 0:
            raise ValueError(f"No scenario pairs found in data_dir={args.data_dir}")

        dataset = MultiScenarioBehaviorDataset(
            scenario_pairs=scenario_pairs,
            min_seq_len=args.min_seq_len,
            include_vehicles_in_graph=not args.no_vehicle_graph,
            cache_labels=not args.no_cache_labels,
        )
        return dataset

    if args.ped_csv:
        dataset = PedestrianBehaviorDataset(
            ped_csv_path=args.ped_csv,
            veh_csv_path=args.veh_csv,
            min_seq_len=args.min_seq_len,
            include_vehicles_in_graph=not args.no_vehicle_graph,
            cache_labels=not args.no_cache_labels,
        )
        return dataset

    raise ValueError("You must provide either --data_dir or --ped_csv")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "x_seq": [x.to(device) for x in batch["x_seq"]],
        "graph_node_feats": [x.to(device) for x in batch["graph_node_feats"]],
        "edge_index": [x.to(device) for x in batch["edge_index"]],
        "edge_attr": [x.to(device) for x in batch["edge_attr"]],
        "target_index": batch["target_index"].to(device),
        "y": batch["y"].to(device),
        "meta": batch["meta"],
    }


def aggregate_predictions(
    records: List[Dict],
    agg: str = "mean",
) -> Dict[int, np.ndarray]:
    """
    records: list of dict with keys
      - ped_id
      - probs: np.ndarray shape (4,)
      - frame_end
      - confidence
    returns:
      dict[ped_id] = np.ndarray shape (4,)
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[int(r["ped_id"])].append(r)

    result: Dict[int, np.ndarray] = {}

    for ped_id, items in grouped.items():
        if agg == "mean":
            probs = np.stack([x["probs"] for x in items], axis=0).mean(axis=0)

        elif agg == "max_conf":
            best_item = max(items, key=lambda x: x["confidence"])
            probs = best_item["probs"]

        elif agg == "latest":
            best_item = max(items, key=lambda x: x["frame_end"])
            probs = best_item["probs"]

        else:
            raise ValueError(f"Unsupported agg={agg}")

        probs = np.asarray(probs, dtype=np.float32)
        s = probs.sum()
        if s <= 0:
            probs = np.ones(4, dtype=np.float32) / 4.0
        else:
            probs = probs / s

        result[ped_id] = probs

    return result


def export_json(mode_probs: Dict[int, np.ndarray], output_json: str) -> None:
    payload = {
        str(pid): probs.tolist()
        for pid, probs in mode_probs.items()
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_csv(mode_probs: Dict[int, np.ndarray], output_csv: str) -> None:
    import csv

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ped_id"] + MODE_NAMES + ["pred_mode", "pred_mode_name"])
        for pid, probs in sorted(mode_probs.items(), key=lambda x: x[0]):
            pred_idx = int(np.argmax(probs))
            writer.writerow(
                [pid] + probs.tolist() + [pred_idx, MODE_NAMES[pred_idx]]
            )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    print(f"[Info] Loaded checkpoint from: {args.checkpoint}")
    if "epoch" in ckpt:
        print(f"[Info] Checkpoint epoch: {ckpt['epoch']}")

    dataset = build_dataset(args)
    if hasattr(dataset, "summary"):
        print("[Info] Dataset summary:")
        print(dataset.summary())

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot run inference.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=behavior_collate_fn,
    )

    window_records: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            probs_batch = model.predict_proba(batch).detach().cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                probs = probs_batch[i]
                confidence = float(np.max(probs))
                ped_id = int(meta["ped_id"])
                frame_end = int(meta["frame_end"])

                window_records.append({
                    "ped_id": ped_id,
                    "frame_end": frame_end,
                    "confidence": confidence,
                    "probs": probs,
                    "label_name": meta.get("label_name"),
                })

    mode_probs = aggregate_predictions(window_records, agg=args.agg)

    export_json(mode_probs, args.output_json)
    print(f"[Info] Saved JSON to: {args.output_json}")

    if args.output_csv:
        export_csv(mode_probs, args.output_csv)
        print(f"[Info] Saved CSV to: {args.output_csv}")

    print("[Info] Example outputs:")
    shown = 0
    for pid, probs in sorted(mode_probs.items(), key=lambda x: x[0]):
        pred_idx = int(np.argmax(probs))
        print(
            f"  ped_id={pid}, probs={np.round(probs, 4)}, "
            f"pred={pred_idx} ({MODE_NAMES[pred_idx]})"
        )
        shown += 1
        if shown >= 10:
            break

    print("[Done] You can now pass this dictionary into MultiModalGSFM(mode_probabilities=...).")


if __name__ == "__main__":
    main()