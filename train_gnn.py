from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from gnn_dataset import (
    PedestrianBehaviorDataset,
    MultiScenarioBehaviorDataset,
    behavior_collate_fn,
    discover_scenario_pairs,
)
from gnn_model import build_behavior_gnn

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    f1s = []

    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return float(np.mean(f1s))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ============================================================
# Dataset utilities
# ============================================================

def build_dataset_from_args(args: argparse.Namespace):
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


def split_dataset(
    dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_labels_from_subset(subset: Subset) -> List[int]:
    labels = []
    for idx in subset.indices:
        item = subset.dataset[idx]
        labels.append(int(item.y.item()))
    return labels


def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for y in labels:
        counts[y] += 1.0

    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.mean()  # normalize around 1

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Train / Eval loops
# ============================================================

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


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0

    all_true = []
    all_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(batch)          # [B, 4]
            loss = criterion(logits, batch["y"])

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = batch["y"].size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        preds = torch.argmax(logits, dim=-1)

        all_true.append(batch["y"].detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    if total_count == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": np.zeros((4, 4), dtype=np.int64),
        }

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    avg_loss = total_loss / total_count
    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes=4)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)

    return {
        "loss": float(avg_loss),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


# ============================================================
# Saving
# ============================================================

def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: Dict,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)


def save_training_summary(
    save_dir: str,
    history: List[Dict],
    best_epoch: int,
    best_metric: float,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "training_history.json")

    serializable_history = []
    for row in history:
        row_copy = dict(row)
        if isinstance(row_copy.get("train_confusion_matrix"), np.ndarray):
            row_copy["train_confusion_matrix"] = row_copy["train_confusion_matrix"].tolist()
        if isinstance(row_copy.get("val_confusion_matrix"), np.ndarray):
            row_copy["val_confusion_matrix"] = row_copy["val_confusion_matrix"].tolist()
        serializable_history.append(row_copy)

    payload = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "history": serializable_history,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BehaviorGNN for pedestrian behavior classification")

    # data input
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing multiple scenario CSVs")
    parser.add_argument("--ped_csv", type=str, default=None,
                        help="Single pedestrian CSV path")
    parser.add_argument("--veh_csv", type=str, default=None,
                        help="Single vehicle CSV path")

    # dataset
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--no_vehicle_graph", action="store_true")
    parser.add_argument("--no_cache_labels", action="store_true")

    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--temporal_layers", type=int, default=1)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # output
    parser.add_argument("--save_dir", type=str, default="checkpoints/behavior_gnn")
    parser.add_argument("--save_name", type=str, default="best_model.pt")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # -----------------------------
    # Build dataset
    # -----------------------------
    dataset = build_dataset_from_args(args)

    if hasattr(dataset, "summary"):
        print("[Info] Dataset summary:")
        print(dataset.summary())

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check labels, CSV paths, or preprocessing.")

    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Info] Train samples: {len(train_set)}")
    print(f"[Info] Val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=behavior_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=behavior_collate_fn,
    )

    # -----------------------------
    # Build model
    # -----------------------------
    model = build_behavior_gnn(
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
    ).to(device)

    # -----------------------------
    # Loss
    # -----------------------------
    if args.use_class_weights:
        train_labels = get_labels_from_subset(train_set)
        class_weights = compute_class_weights(train_labels, num_classes=4).to(device)
        print(f"[Info] Class weights: {class_weights.detach().cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------
    # Train loop
    # -----------------------------
    best_val_macro_f1 = -1.0
    best_epoch = -1
    history = []

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, args.save_name)
    last_model_path = os.path.join(save_dir, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_confusion_matrix": train_metrics["confusion_matrix"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                args=args,
            )
            print(f"[Info] Saved new best model to {best_model_path}")

    # save last model
    save_checkpoint(
        save_path=last_model_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        val_metrics=history[-1] if history else {},
        args=args,
    )
    print(f"[Info] Saved last model to {last_model_path}")

    save_training_summary(
        save_dir=save_dir,
        history=history,
        best_epoch=best_epoch,
        best_metric=best_val_macro_f1,
    )

    print(f"[Done] Best epoch: {best_epoch}, best val macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()