from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PED_MODE_COLUMNS = ["aggressive", "regular", "cautious", "following"]


@dataclass
class PedMetrics:
    ped_id: int
    n_overlap_frames: int
    frame_start: int
    frame_end: int
    ade: float
    fde: float
    rmse_pos: float
    mae_x: float
    mae_y: float
    speed_mae: float
    speed_rmse: float
    path_length_gt: float
    path_length_sim: float
    path_length_abs_error: float
    final_speed_gt: float
    final_speed_sim: float
    final_speed_abs_error: float
    min_pedveh_dist_gt: Optional[float]
    min_pedveh_dist_sim: Optional[float]
    min_pedveh_dist_abs_error: Optional[float]
    collision_gt: Optional[bool]
    collision_sim: Optional[bool]
    collision_match: Optional[bool]
    pred_mode: Optional[int] = None
    pred_mode_name: Optional[str] = None
    prob_aggressive: Optional[float] = None
    prob_regular: Optional[float] = None
    prob_cautious: Optional[float] = None
    prob_following: Optional[float] = None


@dataclass
class AggregateMetrics:
    num_gt_pedestrians: int
    num_sim_pedestrians: int
    num_evaluated_pedestrians: int
    num_missing_in_sim: int
    num_missing_in_gt: int
    mean_ade: float
    median_ade: float
    mean_fde: float
    median_fde: float
    mean_rmse_pos: float
    mean_speed_mae: float
    mean_speed_rmse: float
    mean_path_length_abs_error: float
    mean_min_pedveh_dist_abs_error: Optional[float]
    collision_rate_gt: Optional[float]
    collision_rate_sim: Optional[float]
    collision_match_rate: Optional[float]
    frame_weighted_ade: float
    frame_weighted_rmse_pos: float


# ---------------------------
# Loading helpers
# ---------------------------

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth_ped_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "x_est" in df.columns:
        df = df.rename(columns={"x_est": "x", "y_est": "y", "vx_est": "vx", "vy_est": "vy"})
    required = {"id", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground-truth pedestrian CSV missing columns: {sorted(missing)}")
    if "vx" not in df.columns or "vy" not in df.columns:
        df = compute_velocities_from_positions(df, id_col="id")
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def load_ground_truth_veh_csv(path: str | Path | None) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "x_est" in df.columns:
        df = df.rename(columns={"x_est": "x", "y_est": "y", "psi_est": "psi", "vel_est": "vel"})
    required = {"id", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground-truth vehicle CSV missing columns: {sorted(missing)}")
    if "vel" not in df.columns:
        df = compute_vehicle_speed(df)
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def load_sim_ped_json(path: str | Path) -> pd.DataFrame:
    payload = load_json(path)
    rows = []
    for ped_id, entries in payload.items():
        pid = int(ped_id)
        for row in entries:
            rows.append({
                "id": pid,
                "frame": int(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "vx": float(row.get("vx", np.nan)),
                "vy": float(row.get("vy", np.nan)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No pedestrian simulation rows found in {path}")
    if df["vx"].isna().any() or df["vy"].isna().any():
        df = compute_velocities_from_positions(df, id_col="id")
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def load_sim_veh_json(path: str | Path | None) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    payload = load_json(path)
    rows = []
    for veh_id, entries in payload.items():
        vid = int(veh_id)
        for row in entries:
            rows.append({
                "id": vid,
                "frame": int(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "psi": float(row.get("psi", np.nan)),
                "vel": float(row.get("vel", np.nan)),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df["vel"].isna().any():
        df = compute_vehicle_speed(df)
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def load_mode_probabilities_csv(path: str | Path | None) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ped_id" not in df.columns:
        raise ValueError(f"Mode probability CSV must contain ped_id column: {path}")
    return df


# ---------------------------
# Data utilities
# ---------------------------

def compute_velocities_from_positions(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    out = df.copy().sort_values([id_col, "frame"]).reset_index(drop=True)
    out["vx"] = out.groupby(id_col)["x"].diff()
    out["vy"] = out.groupby(id_col)["y"].diff()
    out["vx"] = out.groupby(id_col)["vx"].transform(lambda s: s.fillna(0.0))
    out["vy"] = out.groupby(id_col)["vy"].transform(lambda s: s.fillna(0.0))
    return out


def compute_vehicle_speed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["id", "frame"]).reset_index(drop=True)
    dx = out.groupby("id")["x"].diff().fillna(0.0)
    dy = out.groupby("id")["y"].diff().fillna(0.0)
    out["vel"] = np.sqrt(dx.to_numpy() ** 2 + dy.to_numpy() ** 2)
    return out


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def speed_from_vxy(df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(df["vx"].to_numpy() ** 2 + df["vy"].to_numpy() ** 2)


def path_length(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    dx = np.diff(df["x"].to_numpy())
    dy = np.diff(df["y"].to_numpy())
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def min_pedveh_distance(ped_track: pd.DataFrame, veh_df: Optional[pd.DataFrame]) -> Optional[float]:
    if veh_df is None or veh_df.empty or ped_track.empty:
        return None
    frames = ped_track["frame"].unique()
    veh_sub = veh_df[veh_df["frame"].isin(frames)]
    if veh_sub.empty:
        return None

    ped_by_frame = {int(f): grp for f, grp in ped_track.groupby("frame")}
    min_dist = math.inf
    found = False
    for frame, veh_grp in veh_sub.groupby("frame"):
        ped_grp = ped_by_frame.get(int(frame))
        if ped_grp is None or ped_grp.empty:
            continue
        pxy = ped_grp[["x", "y"]].to_numpy()
        vxy = veh_grp[["x", "y"]].to_numpy()
        diff = pxy[:, None, :] - vxy[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        cur = float(np.min(dists))
        min_dist = min(min_dist, cur)
        found = True
    return min_dist if found else None


def make_collision_flag(min_dist: Optional[float], threshold: float) -> Optional[bool]:
    if min_dist is None:
        return None
    return bool(min_dist < threshold)


# ---------------------------
# Core evaluation
# ---------------------------

def evaluate_single_run(
    gt_ped_df: pd.DataFrame,
    gt_veh_df: Optional[pd.DataFrame],
    sim_ped_df: pd.DataFrame,
    sim_veh_df: Optional[pd.DataFrame],
    mode_probs_df: Optional[pd.DataFrame],
    collision_threshold: float,
) -> Tuple[pd.DataFrame, AggregateMetrics, pd.DataFrame]:
    gt_ids = set(gt_ped_df["id"].unique().tolist())
    sim_ids = set(sim_ped_df["id"].unique().tolist())
    common_ids = sorted(gt_ids & sim_ids)

    per_ped_rows: List[PedMetrics] = []
    summary_rows: List[Dict[str, object]] = []

    mode_lookup = {}
    if mode_probs_df is not None:
        mode_lookup = {int(r["ped_id"]): r for _, r in mode_probs_df.iterrows()}

    weighted_abs_pos_sum = 0.0
    weighted_sq_pos_sum = 0.0
    total_frames = 0

    for ped_id in common_ids:
        gt_track = gt_ped_df[gt_ped_df["id"] == ped_id].copy()
        sim_track = sim_ped_df[sim_ped_df["id"] == ped_id].copy()
        merged = gt_track.merge(sim_track, on=["id", "frame"], suffixes=("_gt", "_sim"))

        if merged.empty:
            continue

        dx = merged["x_sim"].to_numpy() - merged["x_gt"].to_numpy()
        dy = merged["y_sim"].to_numpy() - merged["y_gt"].to_numpy()
        dist = np.sqrt(dx * dx + dy * dy)

        gt_speed = np.sqrt(merged["vx_gt"].to_numpy() ** 2 + merged["vy_gt"].to_numpy() ** 2)
        sim_speed = np.sqrt(merged["vx_sim"].to_numpy() ** 2 + merged["vy_sim"].to_numpy() ** 2)
        speed_err = sim_speed - gt_speed

        ade = float(np.mean(dist))
        fde = float(dist[-1])
        rmse_pos = float(np.sqrt(np.mean(dist ** 2)))
        mae_x = float(np.mean(np.abs(dx)))
        mae_y = float(np.mean(np.abs(dy)))
        speed_mae = float(np.mean(np.abs(speed_err)))
        speed_rmse = float(np.sqrt(np.mean(speed_err ** 2)))

        path_gt = path_length(gt_track)
        path_sim = path_length(sim_track)
        min_gt = min_pedveh_distance(gt_track, gt_veh_df)
        min_sim = min_pedveh_distance(sim_track, sim_veh_df)
        coll_gt = make_collision_flag(min_gt, collision_threshold)
        coll_sim = make_collision_flag(min_sim, collision_threshold)
        coll_match = None if coll_gt is None or coll_sim is None else bool(coll_gt == coll_sim)
        min_dist_err = None if min_gt is None or min_sim is None else abs(min_sim - min_gt)

        pred_mode = None
        pred_mode_name = None
        p_aggr = p_reg = p_caut = p_foll = None
        if ped_id in mode_lookup:
            row = mode_lookup[ped_id]
            pred_mode = int(row["pred_mode"]) if "pred_mode" in row else None
            pred_mode_name = row.get("pred_mode_name") if hasattr(row, "get") else row["pred_mode_name"] if "pred_mode_name" in row else None
            p_aggr = float(row["aggressive"]) if "aggressive" in row else None
            p_reg = float(row["regular"]) if "regular" in row else None
            p_caut = float(row["cautious"]) if "cautious" in row else None
            p_foll = float(row["following"]) if "following" in row else None

        metrics = PedMetrics(
            ped_id=int(ped_id),
            n_overlap_frames=int(len(merged)),
            frame_start=int(merged["frame"].min()),
            frame_end=int(merged["frame"].max()),
            ade=ade,
            fde=fde,
            rmse_pos=rmse_pos,
            mae_x=mae_x,
            mae_y=mae_y,
            speed_mae=speed_mae,
            speed_rmse=speed_rmse,
            path_length_gt=float(path_gt),
            path_length_sim=float(path_sim),
            path_length_abs_error=float(abs(path_sim - path_gt)),
            final_speed_gt=float(gt_speed[-1]),
            final_speed_sim=float(sim_speed[-1]),
            final_speed_abs_error=float(abs(sim_speed[-1] - gt_speed[-1])),
            min_pedveh_dist_gt=min_gt,
            min_pedveh_dist_sim=min_sim,
            min_pedveh_dist_abs_error=min_dist_err,
            collision_gt=coll_gt,
            collision_sim=coll_sim,
            collision_match=coll_match,
            pred_mode=pred_mode,
            pred_mode_name=pred_mode_name,
            prob_aggressive=p_aggr,
            prob_regular=p_reg,
            prob_cautious=p_caut,
            prob_following=p_foll,
        )
        per_ped_rows.append(metrics)

        summary_rows.append({
            "ped_id": int(ped_id),
            "n_overlap_frames": int(len(merged)),
            "ade": ade,
            "fde": fde,
            "rmse_pos": rmse_pos,
        })

        weighted_abs_pos_sum += float(np.sum(dist))
        weighted_sq_pos_sum += float(np.sum(dist ** 2))
        total_frames += int(len(merged))

    per_ped_df = pd.DataFrame([asdict(r) for r in per_ped_rows]).sort_values("ped_id") if per_ped_rows else pd.DataFrame()

    if per_ped_df.empty:
        raise ValueError("No overlapping pedestrian trajectories found between simulation output and ground truth.")

    min_dist_err_series = per_ped_df["min_pedveh_dist_abs_error"].dropna()
    coll_gt_series = per_ped_df["collision_gt"].dropna()
    coll_sim_series = per_ped_df["collision_sim"].dropna()
    coll_match_series = per_ped_df["collision_match"].dropna()

    agg = AggregateMetrics(
        num_gt_pedestrians=int(len(gt_ids)),
        num_sim_pedestrians=int(len(sim_ids)),
        num_evaluated_pedestrians=int(len(per_ped_df)),
        num_missing_in_sim=int(len(gt_ids - sim_ids)),
        num_missing_in_gt=int(len(sim_ids - gt_ids)),
        mean_ade=float(per_ped_df["ade"].mean()),
        median_ade=float(per_ped_df["ade"].median()),
        mean_fde=float(per_ped_df["fde"].mean()),
        median_fde=float(per_ped_df["fde"].median()),
        mean_rmse_pos=float(per_ped_df["rmse_pos"].mean()),
        mean_speed_mae=float(per_ped_df["speed_mae"].mean()),
        mean_speed_rmse=float(per_ped_df["speed_rmse"].mean()),
        mean_path_length_abs_error=float(per_ped_df["path_length_abs_error"].mean()),
        mean_min_pedveh_dist_abs_error=(float(min_dist_err_series.mean()) if not min_dist_err_series.empty else None),
        collision_rate_gt=(float(coll_gt_series.astype(float).mean()) if not coll_gt_series.empty else None),
        collision_rate_sim=(float(coll_sim_series.astype(float).mean()) if not coll_sim_series.empty else None),
        collision_match_rate=(float(coll_match_series.astype(float).mean()) if not coll_match_series.empty else None),
        frame_weighted_ade=float(weighted_abs_pos_sum / max(total_frames, 1)),
        frame_weighted_rmse_pos=float(math.sqrt(weighted_sq_pos_sum / max(total_frames, 1))),
    )

    missing_df = pd.DataFrame({
        "missing_in_sim": sorted(gt_ids - sim_ids) + [None] * max(0, len(sim_ids - gt_ids) - len(gt_ids - sim_ids)),
        "missing_in_gt": sorted(sim_ids - gt_ids) + [None] * max(0, len(gt_ids - sim_ids) - len(sim_ids - gt_ids)),
    })

    return per_ped_df, agg, missing_df


# ---------------------------
# Reporting helpers
# ---------------------------

def compare_runs(hybrid: AggregateMetrics, baseline: AggregateMetrics) -> Dict[str, float]:
    def pct_improve(base: float, new: float) -> float:
        if base == 0:
            return 0.0
        return 100.0 * (base - new) / base

    out = {
        "ade_improvement_pct": pct_improve(baseline.mean_ade, hybrid.mean_ade),
        "fde_improvement_pct": pct_improve(baseline.mean_fde, hybrid.mean_fde),
        "rmse_pos_improvement_pct": pct_improve(baseline.mean_rmse_pos, hybrid.mean_rmse_pos),
        "speed_mae_improvement_pct": pct_improve(baseline.mean_speed_mae, hybrid.mean_speed_mae),
        "path_length_abs_error_improvement_pct": pct_improve(
            baseline.mean_path_length_abs_error, hybrid.mean_path_length_abs_error
        ),
    }
    if hybrid.mean_min_pedveh_dist_abs_error is not None and baseline.mean_min_pedveh_dist_abs_error is not None:
        out["min_pedveh_dist_abs_error_improvement_pct"] = pct_improve(
            baseline.mean_min_pedveh_dist_abs_error, hybrid.mean_min_pedveh_dist_abs_error
        )
    return out


def mode_summary(mode_probs_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if mode_probs_df is None or mode_probs_df.empty:
        return {}
    summary: Dict[str, object] = {
        "num_pedestrians_with_probs": int(len(mode_probs_df)),
    }
    if "pred_mode_name" in mode_probs_df.columns:
        counts = mode_probs_df["pred_mode_name"].value_counts().to_dict()
        summary["predicted_mode_counts"] = counts
    for col in PED_MODE_COLUMNS:
        if col in mode_probs_df.columns:
            summary[f"mean_prob_{col}"] = float(mode_probs_df[col].mean())
    return summary


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid pedestrian simulation outputs against ground truth, optionally versus a baseline."
    )
    parser.add_argument("--run_summary", type=str, required=True, help="Path to run_summary.json from hybrid pipeline")
    parser.add_argument("--mode_probs_csv", type=str, required=True, help="Path to mode_probabilities.csv")
    parser.add_argument("--pedestrian_output_json", type=str, required=True, help="Path to pedestrian_output.json")
    parser.add_argument("--vehicle_output_json", type=str, default=None, help="Path to vehicle_output.json")
    parser.add_argument("--collision_threshold", type=float, default=1.0, help="Distance threshold used to mark ped-veh collision/risky overlap")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Where to save metrics tables and summary json")

    parser.add_argument("--baseline_pedestrian_output_json", type=str, default=None, help="Optional baseline pedestrian output JSON (e.g. Social Force)")
    parser.add_argument("--baseline_vehicle_output_json", type=str, default=None, help="Optional baseline vehicle output JSON")
    parser.add_argument("--baseline_name", type=str, default="baseline_sfm", help="Name used in baseline comparison outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    run_summary = load_json(args.run_summary)
    gt_ped_csv = run_summary["ped_csv"]
    gt_veh_csv = run_summary.get("veh_csv")

    gt_ped_df = load_ground_truth_ped_csv(gt_ped_csv)
    gt_veh_df = load_ground_truth_veh_csv(gt_veh_csv)

    hybrid_ped_df = load_sim_ped_json(args.pedestrian_output_json)
    hybrid_veh_df = load_sim_veh_json(args.vehicle_output_json)
    mode_probs_df = load_mode_probabilities_csv(args.mode_probs_csv)

    per_ped_df, agg, missing_df = evaluate_single_run(
        gt_ped_df=gt_ped_df,
        gt_veh_df=gt_veh_df,
        sim_ped_df=hybrid_ped_df,
        sim_veh_df=hybrid_veh_df,
        mode_probs_df=mode_probs_df,
        collision_threshold=args.collision_threshold,
    )

    per_ped_path = Path(args.output_dir) / "per_pedestrian_metrics.csv"
    per_ped_df.to_csv(per_ped_path, index=False)
    missing_path = Path(args.output_dir) / "id_coverage.csv"
    missing_df.to_csv(missing_path, index=False)

    summary = {
        "run_summary": run_summary,
        "hybrid_metrics": asdict(agg),
        "mode_probability_summary": mode_summary(mode_probs_df),
        "files": {
            "per_pedestrian_metrics_csv": str(per_ped_path),
            "id_coverage_csv": str(missing_path),
        },
    }

    baseline_comparison = None
    if args.baseline_pedestrian_output_json:
        base_ped_df = load_sim_ped_json(args.baseline_pedestrian_output_json)
        base_veh_df = load_sim_veh_json(args.baseline_vehicle_output_json)
        base_per_ped_df, base_agg, _ = evaluate_single_run(
            gt_ped_df=gt_ped_df,
            gt_veh_df=gt_veh_df,
            sim_ped_df=base_ped_df,
            sim_veh_df=base_veh_df,
            mode_probs_df=None,
            collision_threshold=args.collision_threshold,
        )
        base_per_ped_path = Path(args.output_dir) / f"per_pedestrian_metrics_{args.baseline_name}.csv"
        base_per_ped_df.to_csv(base_per_ped_path, index=False)
        baseline_comparison = {
            "baseline_name": args.baseline_name,
            "baseline_metrics": asdict(base_agg),
            "improvement_vs_baseline_pct": compare_runs(agg, base_agg),
            "baseline_per_pedestrian_metrics_csv": str(base_per_ped_path),
        }
        summary["baseline_comparison"] = baseline_comparison

    summary_path = Path(args.output_dir) / "evaluation_summary.json"
    save_json(summary, summary_path)

    # Console report
    print("\n=== Hybrid Evaluation Summary ===")
    print(f"Evaluated pedestrians: {agg.num_evaluated_pedestrians}")
    print(f"Mean ADE:  {agg.mean_ade:.4f}")
    print(f"Mean FDE:  {agg.mean_fde:.4f}")
    print(f"Mean RMSE: {agg.mean_rmse_pos:.4f}")
    print(f"Mean speed MAE: {agg.mean_speed_mae:.4f}")
    print(f"Frame-weighted ADE: {agg.frame_weighted_ade:.4f}")
    if agg.mean_min_pedveh_dist_abs_error is not None:
        print(f"Mean min ped-veh distance abs error: {agg.mean_min_pedveh_dist_abs_error:.4f}")
    if agg.collision_rate_gt is not None:
        print(f"Collision rate (GT):  {agg.collision_rate_gt:.3f}")
    if agg.collision_rate_sim is not None:
        print(f"Collision rate (SIM): {agg.collision_rate_sim:.3f}")
    if agg.collision_match_rate is not None:
        print(f"Collision match rate: {agg.collision_match_rate:.3f}")

    if baseline_comparison is not None:
        print(f"\n=== Comparison vs {args.baseline_name} ===")
        for k, v in baseline_comparison["improvement_vs_baseline_pct"].items():
            print(f"{k}: {v:+.2f}%")

    print(f"\nSaved evaluation summary to: {summary_path}")
    print(f"Saved per-pedestrian metrics to: {per_ped_path}")


if __name__ == "__main__":
    main()
