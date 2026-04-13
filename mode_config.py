"""
Mode-specific parameter configuration for Multi-Modal GSFM.

This file provides:
1. Global mode definitions
2. Base GSFM parameters
3. Mode-specific parameter overrides
4. Helper functions to fetch one or all parameter sets
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

# ---------------------------------------------------------------------
# Global mode definitions
# ---------------------------------------------------------------------

MODE_NAMES: List[str] = ["aggressive", "regular", "cautious", "following"]
NUM_MODES: int = len(MODE_NAMES)

MODE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(MODE_NAMES)}
INDEX_TO_MODE: Dict[int, str] = {idx: name for idx, name in enumerate(MODE_NAMES)}

# ---------------------------------------------------------------------
# Base GSFM parameters
# These are the default parameters used when no behavior-specific
# modification is applied.
#
# Note:
# - The key names here must match what multimodal_gsfm.py reads.
# - Values below are reasonable initialization values, not final tuned ones.
# ---------------------------------------------------------------------

BASE_PARAMS: Dict[str, float] = {
    # core dynamics
    "mass": 60.0,
    "tau": 0.5,
    "ped_des_speed": 1.34,

    # pedestrian-pedestrian repulsive force
    "ped_ped_range_repul": 3.0,
    "ped_ped_strength_repul": 4.0,
    "anisotropy_lambda_repul": 0.35,

    # pedestrian-pedestrian navigation / avoidance force
    "ped_ped_range_navig": 4.5,
    "ped_ped_strength_navig": 3.0,
    "anisotropy_lambda_navig": 0.8,

    # pedestrian-vehicle interaction force
    "ped_veh_strength": 8.0,
    "ped_veh_decay": 0.7,
    "anisotropy_lambda_p2v": 0.6,

    # blending between ped-ped and ped-veh response
    "ped_single_alpha": 0.65,
    "ped_group_alpha": 0.85,

    # conflict / game-theoretic logic
    "TTC_threshold": 1.0,
    "GT_weight": 0.25,
    "spd_yield_sigma": 0.60,
    "spd_go_sigma": 1.20,

    # crowd-following logic
    "crowd_range_fo": 3.5,
    "crowd_anisotropy_fo": 45.0,
}

# ---------------------------------------------------------------------
# Mode-specific overrides
#
# Interpretation:
# aggressive:
#   - higher desired speed
#   - weaker yielding
#   - slightly smaller ped-veh repulsion
#   - stronger "go" tendency
#
# regular:
#   - base behavior, close to default parameters
#
# cautious:
#   - lower desired speed
#   - stronger yielding
#   - larger ped-veh repulsion
#   - more conservative TTC sensitivity
#
# following:
#   - stronger crowd following
#   - more group-oriented alpha
#   - less individualized behavior
# ---------------------------------------------------------------------

MODE_OVERRIDES: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "ped_des_speed": 1.55,
        "ped_veh_strength": 6.5,
        "ped_veh_decay": 0.60,
        "ped_single_alpha": 0.55,
        "ped_group_alpha": 0.90,
        "TTC_threshold": 0.85,
        "GT_weight": 0.35,
        "spd_yield_sigma": 0.80,
        "spd_go_sigma": 1.35,
        "crowd_range_fo": 3.0,
        "crowd_anisotropy_fo": 35.0,
    },

    "regular": {
        # keep close to base
        "ped_des_speed": 1.34,
        "ped_veh_strength": 8.0,
        "ped_veh_decay": 0.70,
        "ped_single_alpha": 0.65,
        "ped_group_alpha": 0.85,
        "TTC_threshold": 1.00,
        "GT_weight": 0.25,
        "spd_yield_sigma": 0.60,
        "spd_go_sigma": 1.20,
        "crowd_range_fo": 3.5,
        "crowd_anisotropy_fo": 45.0,
    },

    "cautious": {
        "ped_des_speed": 1.05,
        "ped_veh_strength": 11.0,
        "ped_veh_decay": 0.85,
        "ped_single_alpha": 0.80,
        "ped_group_alpha": 0.88,
        "TTC_threshold": 1.30,
        "GT_weight": 0.18,
        "spd_yield_sigma": 0.40,
        "spd_go_sigma": 1.05,
        "crowd_range_fo": 4.2,
        "crowd_anisotropy_fo": 55.0,
    },

    "following": {
        "ped_des_speed": 1.20,
        "ped_veh_strength": 7.5,
        "ped_veh_decay": 0.72,
        "ped_single_alpha": 0.45,
        "ped_group_alpha": 0.95,
        "TTC_threshold": 1.05,
        "GT_weight": 0.22,
        "spd_yield_sigma": 0.55,
        "spd_go_sigma": 1.10,
        "crowd_range_fo": 5.0,
        "crowd_anisotropy_fo": 70.0,
    },
}

# ---------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------

_REQUIRED_KEYS = {
    "mass",
    "tau",
    "ped_des_speed",
    "ped_ped_range_repul",
    "ped_ped_strength_repul",
    "anisotropy_lambda_repul",
    "ped_ped_range_navig",
    "ped_ped_strength_navig",
    "anisotropy_lambda_navig",
    "ped_veh_strength",
    "ped_veh_decay",
    "anisotropy_lambda_p2v",
    "ped_single_alpha",
    "ped_group_alpha",
    "TTC_threshold",
    "GT_weight",
    "spd_yield_sigma",
    "spd_go_sigma",
    "crowd_range_fo",
    "crowd_anisotropy_fo",
}


def _validate_params(params: Dict[str, float], mode_name: str) -> None:
    missing = _REQUIRED_KEYS - set(params.keys())
    if missing:
        raise KeyError(
            f"Mode '{mode_name}' is missing required parameter keys: {sorted(missing)}"
        )

    if params["mass"] <= 0:
        raise ValueError(f"Mode '{mode_name}': mass must be > 0")
    if params["tau"] <= 0:
        raise ValueError(f"Mode '{mode_name}': tau must be > 0")
    if params["ped_des_speed"] <= 0:
        raise ValueError(f"Mode '{mode_name}': ped_des_speed must be > 0")
    if params["ped_single_alpha"] < 0 or params["ped_single_alpha"] > 1:
        raise ValueError(f"Mode '{mode_name}': ped_single_alpha must be in [0, 1]")
    if params["ped_group_alpha"] < 0 or params["ped_group_alpha"] > 1:
        raise ValueError(f"Mode '{mode_name}': ped_group_alpha must be in [0, 1]")
    if params["TTC_threshold"] <= 0:
        raise ValueError(f"Mode '{mode_name}': TTC_threshold must be > 0")
    if params["spd_yield_sigma"] <= 0:
        raise ValueError(f"Mode '{mode_name}': spd_yield_sigma must be > 0")
    if params["spd_go_sigma"] <= 0:
        raise ValueError(f"Mode '{mode_name}': spd_go_sigma must be > 0")


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

def get_mode_name(mode_idx: int) -> str:
    if mode_idx not in INDEX_TO_MODE:
        raise IndexError(f"Invalid mode index {mode_idx}. Valid range: [0, {NUM_MODES - 1}]")
    return INDEX_TO_MODE[mode_idx]


def get_mode_index(mode_name: str) -> int:
    if mode_name not in MODE_TO_INDEX:
        raise KeyError(f"Unknown mode name '{mode_name}'. Valid names: {MODE_NAMES}")
    return MODE_TO_INDEX[mode_name]


def get_mode_params(mode: int | str) -> Dict[str, float]:
    """
    Return a full parameter dictionary for one mode.

    Args:
        mode:
            Either an integer mode index or a string mode name.

    Returns:
        Dict[str, float]: merged parameter dictionary
    """
    if isinstance(mode, int):
        mode_name = get_mode_name(mode)
    elif isinstance(mode, str):
        mode_name = mode
        if mode_name not in MODE_OVERRIDES:
            raise KeyError(f"Unknown mode name '{mode_name}'. Valid names: {MODE_NAMES}")
    else:
        raise TypeError("mode must be either int or str")

    params = deepcopy(BASE_PARAMS)
    params.update(MODE_OVERRIDES[mode_name])
    _validate_params(params, mode_name)
    return params


def get_all_mode_params() -> List[Dict[str, float]]:
    """
    Return all mode-specific parameter dictionaries in mode index order.
    """
    return [get_mode_params(i) for i in range(NUM_MODES)]


def get_mode_prior() -> List[float]:
    """
    Optional helper for default prior probabilities.
    """
    return [1.0 / NUM_MODES] * NUM_MODES


# ---------------------------------------------------------------------
# Quick debug / smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Available modes:", MODE_NAMES)
    for i, name in enumerate(MODE_NAMES):
        params = get_mode_params(i)
        print(f"\nMode {i}: {name}")
        for k in sorted(params.keys()):
            print(f"  {k}: {params[k]}")