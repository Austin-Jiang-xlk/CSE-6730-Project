from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from inference_mode_probs import (
    load_model_from_checkpoint,
    build_dataset,
    aggregate_predictions,
    move_batch_to_device,
)
from gnn_dataset import behavior_collate_fn
from torch.utils.data import DataLoader
from multimodal_gsfm import MultiModalGSFM


MODE_NAMES = ["aggressive", "regular", "cautious", "following"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end hybrid pipeline: GNN mode inference + MultiModal GSFM simulation"
    )

    # checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained GNN checkpoint",
    )

    # input scene
    parser.add_argument(
        "--ped_csv",
        type=str,
        required=True,
        help="Pedestrian trajectory csv path",
    )
    parser.add_argument(
        "--veh_csv",
        type=str,
        default=None,
        help="Vehicle trajectory csv path",
    )

    # inference options
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_vehicle_graph", action="store_true")
    parser.add_argument("--no_cache_labels", action="store_true")
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "max_conf", "latest"],
        help="Aggregation rule for window-level predictions per pedestrian",
    )

    # simulation options
    parser.add_argument(
        "--sim_horizon",
        type=int,
        default=None,
        help="Optional simulation horizon override if MultiModalGSFM supports it",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/hybrid_pipeline",
        help="Directory to save inferred probabilities and simulated trajectories",
    )
    parser.add_argument(
        "--save_mode_probs_json",
        action="store_true",
        help="Also save mode probability dictionary as json",
    )
    parser.add_argument(
        "--save_mode_probs_csv",
        action="store_true",
        help="Also save mode probability dictionary as csv",
    )

    return parser.parse_args()


def infer_mode_probabilities(
    checkpoint_path: str,
    ped_csv: str,
    veh_csv: Optional[str],
    min_seq_len: int,
    batch_size: int,
    use_vehicle_graph: bool,
    cache_labels: bool,
    agg: str,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    Run trained GNN on one scene and return:
        dict[ped_id] = np.ndarray shape (4,)
    """
    class _Args:
        pass

    tmp_args = _Args()
    tmp_args.data_dir = None
    tmp_args.ped_csv = ped_csv
    tmp_args.veh_csv = veh_csv
    tmp_args.min_seq_len = min_seq_len
    tmp_args.no_vehicle_graph = not use_vehicle_graph
    tmp_args.no_cache_labels = not cache_labels

    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    dataset = build_dataset(tmp_args)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot run hybrid pipeline.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=behavior_collate_fn,
    )

    window_records = []

    model.eval()
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
                })

    mode_probs = aggregate_predictions(window_records, agg=agg)
    return mode_probs


def save_mode_probs_json(mode_probs: Dict[int, np.ndarray], save_path: str) -> None:
    payload = {str(pid): probs.tolist() for pid, probs in mode_probs.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_mode_probs_csv(mode_probs: Dict[int, np.ndarray], save_path: str) -> None:
    rows = []
    for pid, probs in sorted(mode_probs.items(), key=lambda x: x[0]):
        pred_idx = int(np.argmax(probs))
        rows.append({
            "ped_id": pid,
            "aggressive": float(probs[0]),
            "regular": float(probs[1]),
            "cautious": float(probs[2]),
            "following": float(probs[3]),
            "pred_mode": pred_idx,
            "pred_mode_name": MODE_NAMES[pred_idx],
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def normalize_simulation_output(sim_output):
    """
    Try to normalize output from MultiModalGSFM.simulate(...) into a dict.

    Because the exact return signature may vary, this function handles common cases:
      1. dict
      2. tuple/list of length 2 or more
      3. raw DataFrame / ndarray
    """
    if isinstance(sim_output, dict):
        return sim_output

    if isinstance(sim_output, (tuple, list)):
        out = {}
        if len(sim_output) >= 1:
            out["pedestrian_output"] = sim_output[0]
        if len(sim_output) >= 2:
            out["vehicle_output"] = sim_output[1]
        if len(sim_output) >= 3:
            out["extra_output"] = sim_output[2]
        return out

    return {"pedestrian_output": sim_output}


def save_simulation_outputs(sim_results: Dict, output_dir: str) -> None:
    """
    Save simulation outputs to csv/json depending on object type.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, value in sim_results.items():
        if value is None:
            continue

        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)

        elif isinstance(value, np.ndarray):
            np.save(os.path.join(output_dir, f"{key}.npy"), value)

        elif isinstance(value, dict):
            serializable = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable[str(k)] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    serializable[str(k)] = float(v)
                else:
                    serializable[str(k)] = v
            with open(os.path.join(output_dir, f"{key}.json"), "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)

        elif isinstance(value, list):
            with open(os.path.join(output_dir, f"{key}.json"), "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)

        else:
            with open(os.path.join(output_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(str(value))


def run_multimodal_simulation(
    ped_csv: str,
    veh_csv: Optional[str],
    mode_probabilities: Dict[int, np.ndarray],
    sim_horizon: Optional[int] = None,
):
    """
    Instantiate MultiModalGSFM and run simulation.

    Important:
    This function assumes your MultiModalGSFM supports:
      - constructor argument: mode_probabilities=...
      - a simulate(...) method that accepts ped_csv and veh_csv

    If your actual simulate signature differs, change only this function.
    """
    sim = MultiModalGSFM(mode_probabilities=mode_probabilities)

    # Try the most likely signatures first.
    if sim_horizon is not None:
        try:
            return sim.simulate(ped_csv, veh_csv, sim_horizon=sim_horizon)
        except TypeError:
            pass
        try:
            return sim.simulate(ped_csv_path=ped_csv, veh_csv_path=veh_csv, sim_horizon=sim_horizon)
        except TypeError:
            pass

    try:
        return sim.simulate(ped_csv, veh_csv)
    except TypeError:
        pass

    try:
        return sim.simulate(ped_csv_path=ped_csv, veh_csv_path=veh_csv)
    except TypeError:
        pass

    # Last fallback: some versions may only accept pedestrian input
    try:
        return sim.simulate(ped_csv)
    except TypeError as e:
        raise TypeError(
            "Could not match MultiModalGSFM.simulate(...) signature. "
            "Edit run_multimodal_simulation() to match your actual implementation."
        ) from e


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Step 1: infer mode probabilities
    # ------------------------------------------------------------
    print("[Step 1] Inferring pedestrian mode probabilities ...")
    mode_probs = infer_mode_probabilities(
        checkpoint_path=args.checkpoint,
        ped_csv=args.ped_csv,
        veh_csv=args.veh_csv,
        min_seq_len=args.min_seq_len,
        batch_size=args.batch_size,
        use_vehicle_graph=not args.no_vehicle_graph,
        cache_labels=not args.no_cache_labels,
        agg=args.agg,
        device=device,
    )

    print(f"[Info] Inferred probabilities for {len(mode_probs)} pedestrians")

    if args.save_mode_probs_json:
        json_path = output_dir / "mode_probabilities.json"
        save_mode_probs_json(mode_probs, str(json_path))
        print(f"[Info] Saved mode probabilities JSON to {json_path}")

    if args.save_mode_probs_csv:
        csv_path = output_dir / "mode_probabilities.csv"
        save_mode_probs_csv(mode_probs, str(csv_path))
        print(f"[Info] Saved mode probabilities CSV to {csv_path}")

    # show a few predictions
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

    # ------------------------------------------------------------
    # Step 2: run multimodal GSFM simulation
    # ------------------------------------------------------------
    print("[Step 2] Running MultiModalGSFM simulation ...")
    sim_output = run_multimodal_simulation(
        ped_csv=args.ped_csv,
        veh_csv=args.veh_csv,
        mode_probabilities=mode_probs,
        sim_horizon=args.sim_horizon,
    )

    sim_results = normalize_simulation_output(sim_output)
    save_simulation_outputs(sim_results, str(output_dir))
    print(f"[Info] Saved simulation outputs to {output_dir}")

    # ------------------------------------------------------------
    # Step 3: save lightweight summary
    # ------------------------------------------------------------
    summary = {
        "checkpoint": args.checkpoint,
        "ped_csv": args.ped_csv,
        "veh_csv": args.veh_csv,
        "num_pedestrians_with_mode_probs": len(mode_probs),
        "aggregation": args.agg,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Hybrid pipeline completed.")


if __name__ == "__main__":
    main()