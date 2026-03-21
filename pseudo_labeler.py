"""
pseudo_labeler.py
=================
Rule-based relational pseudo-labeler for 4-mode behavioral classification.

Outcome-Driven Game-Theoretic Classification (v2)
==================================================
Key improvement over v1: instead of classifying based on instantaneous acceleration
at a single mid-frame snapshot, this version tracks the FULL TTC evolution across
the observation window and classifies based on the GAME OUTCOME — who ultimately
wins the right-of-way.

The 4 modes:
  0 - Aggressive:  In conflict + initially disadvantaged (ΔTTC_init ≥ 0) +
                    does NOT yield → pedestrian wins right-of-way (TTC_ped → 0)
  1 - Regular:     No active conflict with vehicle, or in conflict but does not
                    meet the strict causal criteria for Aggressive/Cautious
  2 - Cautious:    In conflict + initially advantaged (ΔTTC_init < 0) +
                    yields significantly → vehicle wins right-of-way (TTC_veh → 0)
  3 - Following:   Crowd follower (spatial proximity + velocity alignment)

Theoretical basis (from Gemini analysis):
  - Aggressive = "逆风翻盘型" (comeback against headwind):
    starts disadvantaged, refuses to yield, ultimately passes first
  - Cautious = "顺风礼让型" (yields despite tailwind):
    starts advantaged, actively decelerates, lets vehicle pass first

Usage:
    from pseudo_labeler import PseudoLabeler
    labeler = PseudoLabeler()
    labels_df = labeler.label_scenario(ped_csv, veh_csv)
"""

import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict


class PseudoLabeler:
    """
    Assigns behavioral mode labels to pedestrians in DUT/CITR trajectory data
    using outcome-driven game-theoretic heuristics grounded in the GSFM framework.
    """

    def __init__(self,
                 # Crowd follower detection (from thesis Section 3.4.1)
                 crowd_range: float = 2.68,
                 crowd_angle_deg: float = 60.0,
                 velocity_alignment_threshold: float = 0.7,
                 # TTC conflict detection (from thesis Section 3.3.1)
                 ttc_threshold: float = 2.6,
                 dist_tol: float = 1.0,
                 # Velocity angle filter for conflict (from thesis Section 4.x)
                 # Only angles between [min_deg, max_deg] count as genuine conflict
                 # 45°~135° filters out parallel co-moving agents, keeps crossing/approaching
                 conflict_angle_min_deg: float = 45.0,
                 conflict_angle_max_deg: float = 135.0,
                 # Game outcome thresholds
                 ttc_convergence: float = 0.8,      # TTC → 0 means TTC < this value
                 ttc_remaining_min: float = 0.5,     # the loser's TTC must stay above this
                 decel_threshold: float = -0.01,       # m/s^2, significant deceleration.  #####IMPORTANT
                 # TTC sampling: how many evenly-spaced frames to sample within window
                 ttc_sample_count: int = 10,
                 # General
                 ped_des_speed: float = 1.44,
                 # Sliding window
                 fps: float = 30.0,
                 window_sec: float = 5.0,
                 stride_sec: float = 1.0):

        self.crowd_range = crowd_range
        self.crowd_angle_deg = crowd_angle_deg
        self.vel_align_thresh = velocity_alignment_threshold
        self.ttc_threshold = ttc_threshold
        self.dist_tol = dist_tol
        self.conflict_angle_min = conflict_angle_min_deg
        self.conflict_angle_max = conflict_angle_max_deg
        self.ttc_convergence = ttc_convergence
        self.ttc_remaining_min = ttc_remaining_min
        self.decel_threshold = decel_threshold
        self.ttc_sample_count = ttc_sample_count
        self.ped_des_speed = ped_des_speed
        self.fps = fps
        self.window_frames = int(window_sec * fps)
        self.stride_frames = int(stride_sec * fps)

    # -----------------------------------------------------------------
    # Velocity angle between two agents
    # -----------------------------------------------------------------
    @staticmethod
    def velocity_angle_deg(v1, v2):
        """
        Compute the angle (degrees) between two velocity vectors.
        Returns value in [0, 180].
          0°   = exactly parallel, same direction
          90°  = perpendicular (crossing)
          180° = exactly head-on
        """
        s1 = np.linalg.norm(v1)
        s2 = np.linalg.norm(v2)
        if s1 < 1e-6 or s2 < 1e-6:
            return 0.0  # stationary agent → no meaningful angle
        cos_a = np.clip(np.dot(v1, v2) / (s1 * s2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_a))

    # -----------------------------------------------------------------
    # Core TTC computation with angle filter
    # (mirrors GSFM.ttc_dual from thesis Eq. 3-12, enhanced with
    #  velocity angle condition from thesis Section 4.x)
    # -----------------------------------------------------------------
    def compute_ttc(self, p_ped, v_ped, p_veh, v_veh):
        """
        Solve the linear TTC system:
            p_ped + Tp * v_ped = p_veh + Tv * v_veh

        Conflict requires conditions:
          1. Approaching: agents must be closing distance (not separating)
          2. Temporal: Tp > 0, Tv > 0, |Tp - Tv| < ttc_threshold
          3. Spatial:  projected intersection points < dist_tol apart
          4. Angular:  velocity angle ∈ [conflict_angle_min, conflict_angle_max]

        Returns (Tp, Tv, conflict_flag)
        """
        # Condition 1: approaching check
        # Relative position vector from ped to veh
        rel_pos = p_veh - p_ped
        # Relative velocity (closing velocity): positive dot product = closing
        rel_vel = v_ped - v_veh  # velocity of ped relative to veh
        closing_rate = np.dot(rel_pos, rel_vel)
        # If closing_rate <= 0, the ped is moving AWAY from veh (or parallel)
        # → they are separating, not approaching → no conflict
        if closing_rate <= 0:
            return np.inf, np.inf, False

        A = np.column_stack((v_ped, -v_veh))
        b = p_veh - p_ped

        if np.linalg.matrix_rank(A) < 2:
            return np.inf, np.inf, False

        try:
            Tp, Tv = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf, np.inf, False

        # Condition 2: temporal — both positive, close together
        if Tp > 0 and Tv > 0 and abs(Tp - Tv) < self.ttc_threshold:
            # Condition 3: spatial — projected points actually meet
            p_int_ped = p_ped + Tp * v_ped
            p_int_veh = p_veh + Tv * v_veh
            if np.linalg.norm(p_int_ped - p_int_veh) < self.dist_tol:
                # Condition 4: velocity angle filter
                angle = self.velocity_angle_deg(v_ped, v_veh)
                if self.conflict_angle_min <= angle <= self.conflict_angle_max:
                    return Tp, Tv, True
        return Tp, Tv, False

    # -----------------------------------------------------------------
    # TTC time-series tracker across an entire window
    # -----------------------------------------------------------------
    def track_ttc_evolution(self, ped_data, veh_data, pid, win_start, win_end):
        """
        Track TTC_ped and TTC_veh evolution across the observation window
        for a given pedestrian against all vehicles.

        Returns dict:
            {
                'has_conflict': bool,
                'conflict_veh_id': int or None,
                'ttc_ped_series': list of (frame, Tp) tuples,
                'ttc_veh_series': list of (frame, Tv) tuples,
                'delta_ttc_init': float or None,   # Tp - Tv at first conflict frame
                'delta_ttc_final': float or None,  # Tp - Tv at last conflict frame
                'ttc_ped_min': float,               # min TTC_ped across window
                'ttc_veh_min': float,               # min TTC_veh across window
                'conflict_frame_count': int,        # how many sampled frames had conflict
            }
        """
        result = {
            'has_conflict': False,
            'conflict_veh_id': None,
            'ttc_ped_series': [],
            'ttc_veh_series': [],
            'delta_ttc_init': None,
            'delta_ttc_final': None,
            'ttc_ped_min': np.inf,
            'ttc_veh_min': np.inf,
            'conflict_frame_count': 0,
            'mean_conflict_angle': None,
        }

        if veh_data.empty:
            return result

        # Sample frames evenly across the window
        all_frames = sorted(ped_data[
            (ped_data['id'] == pid) &
            (ped_data['frame'] >= win_start) &
            (ped_data['frame'] < win_end)
        ]['frame'].unique())

        if len(all_frames) < 3:
            return result

        # Subsample to avoid excessive computation
        step = max(1, len(all_frames) // self.ttc_sample_count)
        sample_frames = all_frames[::step]

        # For each sampled frame, compute TTC against all vehicles
        # Track the vehicle that produces the MOST conflict frames
        veh_conflict_counts = {}

        for frame in sample_frames:
            ped_at_frame = ped_data[
                (ped_data['id'] == pid) &
                (ped_data['frame'] == frame)
            ]
            if ped_at_frame.empty:
                continue

            ped_row = ped_at_frame.iloc[0]
            p_ped = np.array([ped_row['x'], ped_row['y']])
            v_ped = np.array([ped_row['vx'], ped_row['vy']])

            # Find vehicles near this frame (±2 frames tolerance)
            vehs_nearby = veh_data[
                (veh_data['frame'] >= frame - 2) &
                (veh_data['frame'] <= frame + 2)
            ]

            for _, veh_row in vehs_nearby.iterrows():
                if 'veh_vx' not in veh_row:
                    continue

                vid = int(veh_row['id'])
                p_veh = np.array([veh_row['x'], veh_row['y']])
                v_veh = np.array([veh_row['veh_vx'], veh_row['veh_vy']])

                Tp, Tv, conflict = self.compute_ttc(p_ped, v_ped, p_veh, v_veh)

                if conflict:
                    veh_conflict_counts[vid] = veh_conflict_counts.get(vid, 0) + 1

        if not veh_conflict_counts:
            return result

        # Select the primary conflict vehicle (most conflict frames)
        primary_vid = max(veh_conflict_counts, key=veh_conflict_counts.get)
        result['has_conflict'] = True
        result['conflict_veh_id'] = primary_vid

        # Now re-scan ALL sampled frames for the primary conflict vehicle
        # to build the full TTC time series
        ttc_ped_list = []
        ttc_veh_list = []
        conflict_angles = []

        for frame in sample_frames:
            ped_at_frame = ped_data[
                (ped_data['id'] == pid) &
                (ped_data['frame'] == frame)
            ]
            if ped_at_frame.empty:
                continue

            ped_row = ped_at_frame.iloc[0]
            p_ped = np.array([ped_row['x'], ped_row['y']])
            v_ped = np.array([ped_row['vx'], ped_row['vy']])

            # Find this vehicle near this frame
            veh_at_frame = veh_data[
                (veh_data['id'] == primary_vid) &
                (veh_data['frame'] >= frame - 2) &
                (veh_data['frame'] <= frame + 2)
            ]
            if veh_at_frame.empty:
                continue

            veh_row = veh_at_frame.iloc[len(veh_at_frame) // 2]
            if 'veh_vx' not in veh_row:
                continue

            p_veh = np.array([veh_row['x'], veh_row['y']])
            v_veh = np.array([veh_row['veh_vx'], veh_row['veh_vy']])

            Tp, Tv, conflict = self.compute_ttc(p_ped, v_ped, p_veh, v_veh)

            # Only record valid (positive, finite) TTC values
            if Tp > 0 and Tv > 0 and not np.isinf(Tp) and not np.isinf(Tv):
                ttc_ped_list.append((frame, Tp))
                ttc_veh_list.append((frame, Tv))
                if conflict:
                    result['conflict_frame_count'] += 1
                    conflict_angles.append(self.velocity_angle_deg(v_ped, v_veh))

        result['ttc_ped_series'] = ttc_ped_list
        result['ttc_veh_series'] = ttc_veh_list
        if conflict_angles:
            result['mean_conflict_angle'] = float(np.mean(conflict_angles))

        if len(ttc_ped_list) >= 2:
            # Initial ΔTTC: first valid measurement
            result['delta_ttc_init'] = ttc_ped_list[0][1] - ttc_veh_list[0][1]
            # Final ΔTTC: last valid measurement
            result['delta_ttc_final'] = ttc_ped_list[-1][1] - ttc_veh_list[-1][1]
            # Min TTC values (who approached 0?)
            result['ttc_ped_min'] = min(tp for _, tp in ttc_ped_list)
            result['ttc_veh_min'] = min(tv for _, tv in ttc_veh_list)

        # If too few conflict frames relative to total samples, downgrade
        if result['conflict_frame_count'] < 2:
            result['has_conflict'] = False

        return result

    # -----------------------------------------------------------------
    # Outcome-driven game resolution classifier
    # -----------------------------------------------------------------
    def classify_game_outcome(self, ttc_result, speeds, dt):
        """
        Classify behavioral mode based on game-theoretic outcome.

        Decision logic (4 levels):
        ┌─────────────────────────────────────────────────────────────┐
        │ Level 1: Is there a conflict?                              │
        │   NO  → Regular                                            │
        │   YES → proceed to Level 2                                 │
        ├─────────────────────────────────────────────────────────────┤
        │ Level 2: What is the initial game stance?                  │
        │   ΔTTC_init = TTC_ped(t0) - TTC_veh(t0)                   │
        │   ≥ 0 : ped disadvantaged (veh arrives first or tie)       │
        │   < 0 : ped advantaged (ped arrives first)                 │
        ├─────────────────────────────────────────────────────────────┤
        │ Level 3: What is the speed response during conflict?       │
        │   mean_accel ≥ decel_threshold: no significant deceleration│
        │   mean_accel <  decel_threshold: significant deceleration  │
        ├─────────────────────────────────────────────────────────────┤
        │ Level 4: Who wins? (TTC convergence)                       │
        │   TTC_ped → 0 while TTC_veh > 0: ped passes first (wins)  │
        │   TTC_veh → 0 while TTC_ped > 0: veh passes first (loses) │
        │   Neither clear → Regular                                  │
        └─────────────────────────────────────────────────────────────┘

        Aggressive (Mode 0) requires ALL of:
          ✓ has_conflict
          ✓ ΔTTC_init ≥ 0 (initially disadvantaged)
          ✓ no significant deceleration (mean_accel ≥ decel_threshold)
          ✓ TTC_ped_min < convergence AND TTC_veh_min > remaining_min
            (ped ultimately passes first)

        Cautious (Mode 2) requires ALL of:
          ✓ has_conflict
          ✓ ΔTTC_init < 0 (initially advantaged)
          ✓ significant deceleration (mean_accel < decel_threshold)
          ✓ TTC_veh_min < convergence AND TTC_ped_min > remaining_min
            (veh ultimately passes first)

        All other conflict cases → Regular (Mode 1)
        """
        if not ttc_result['has_conflict']:
            return 1, 'regular'

        delta_init = ttc_result['delta_ttc_init']
        ttc_ped_min = ttc_result['ttc_ped_min']
        ttc_veh_min = ttc_result['ttc_veh_min']

        # Need valid TTC evolution data
        if delta_init is None:
            return 1, 'regular'

        # Compute mean acceleration over the window
        if len(speeds) >= 3:
            accels = np.diff(speeds) / dt
            mean_accel = np.mean(accels)
        else:
            mean_accel = 0.0

        # --- Aggressive: 逆风翻盘型 ---
        # Initially disadvantaged + doesn't yield + wins right-of-way
        ped_wins = (ttc_ped_min < self.ttc_convergence and
                    ttc_veh_min > self.ttc_remaining_min)
        if (delta_init >= 0 and
                mean_accel >= self.decel_threshold and
                ped_wins):
            return 0, 'aggressive'

        # --- Cautious: 顺风礼让型 ---
        # Initially advantaged + yields significantly + vehicle wins right-of-way
        veh_wins = (ttc_veh_min < self.ttc_convergence and
                    ttc_ped_min > self.ttc_remaining_min)
        if (delta_init < 0 and
                mean_accel < self.decel_threshold and
                veh_wins):
            return 2, 'cautious'

        # --- All other conflict cases: Regular ---
        return 1, 'regular'

    # -----------------------------------------------------------------
    # Crowd follower detection (mirrors GSFM.is_crowd_follower, Section 3.4.1)
    # -----------------------------------------------------------------
    def is_crowd_follower(self, ego_row, other_rows):
        """
        Check if ego pedestrian has leaders in front within angular cone.
        Returns True if ego is a follower (has leaders ahead with similar velocity).
        """
        ego_pos = np.array([ego_row['x'], ego_row['y']])
        ego_vel = np.array([ego_row['vx'], ego_row['vy']])
        ego_speed = np.linalg.norm(ego_vel)

        if ego_speed < 1e-4:
            return False

        leaders_found = 0
        for _, other in other_rows.iterrows():
            other_pos = np.array([other['x'], other['y']])
            other_vel = np.array([other['vx'], other['vy']])

            vec_to_other = other_pos - ego_pos
            dist = np.linalg.norm(vec_to_other)

            if dist < 1e-3 or dist > self.crowd_range:
                continue

            cos_angle = np.dot(ego_vel, vec_to_other) / (ego_speed * dist + 1e-9)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            if angle_deg > self.crowd_angle_deg:
                continue

            other_speed = np.linalg.norm(other_vel)
            if other_speed < 1e-4:
                continue

            vel_cos = np.dot(ego_vel, other_vel) / (ego_speed * other_speed + 1e-9)
            if vel_cos >= self.vel_align_thresh:
                leaders_found += 1

        return leaders_found > 0

    # -----------------------------------------------------------------
    # Main labeling pipeline for a single scenario
    # -----------------------------------------------------------------
    def label_scenario(self,
                       ped_csv_path: str,
                       veh_csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Label all pedestrians in a scenario with behavioral modes.

        Returns DataFrame with columns:
            [ped_id, frame_start, frame_end, mode, mode_name,
             has_conflict, is_follower, mean_speed, mean_accel,
             delta_ttc_init, delta_ttc_final, ttc_ped_min, ttc_veh_min,
             conflict_veh_id]
        """
        import os
        ped_data = pd.read_csv(ped_csv_path)

        # Handle column name variants
        if 'x_est' in ped_data.columns:
            ped_data = ped_data.rename(columns={
                'x_est': 'x', 'y_est': 'y', 'vx_est': 'vx', 'vy_est': 'vy'
            })

        # Load vehicle data
        if veh_csv_path and os.path.exists(veh_csv_path):
            veh_data = pd.read_csv(veh_csv_path)
            if 'x_est' in veh_data.columns:
                veh_data = veh_data.rename(columns={
                    'x_est': 'x', 'y_est': 'y', 'psi_est': 'psi', 'vel_est': 'vel'
                })
            # Precompute vehicle vx, vy
            if 'veh_vx' not in veh_data.columns and 'psi' in veh_data.columns:
                veh_data['veh_vx'] = veh_data['vel'] * np.cos(veh_data['psi'])
                veh_data['veh_vy'] = veh_data['vel'] * np.sin(veh_data['psi'])
        else:
            veh_data = pd.DataFrame()

        # Compute per-pedestrian speed
        ped_data['speed'] = np.sqrt(ped_data['vx']**2 + ped_data['vy']**2)

        dt = 1.0 / self.fps
        all_labels = []

        ped_ids = ped_data['id'].unique()
        frame_min = int(ped_data['frame'].min())
        frame_max = int(ped_data['frame'].max())

        # Sliding window over the scenario timeline
        for win_start in range(frame_min, frame_max - self.window_frames + 1, self.stride_frames):
            win_end = win_start + self.window_frames

            for pid in ped_ids:
                # Get this pedestrian's data in the window
                mask = (ped_data['id'] == pid) & \
                       (ped_data['frame'] >= win_start) & \
                       (ped_data['frame'] < win_end)
                ped_window = ped_data[mask].sort_values('frame')

                if len(ped_window) < 5:
                    continue

                # ============================================================
                # Level 1: Check crowd follower status (RELATIONAL)
                # ============================================================
                mid_frame = win_start + self.window_frames // 2
                mid_row_candidates = ped_window[
                    (ped_window['frame'] >= mid_frame - 2) &
                    (ped_window['frame'] <= mid_frame + 2)
                ]
                if mid_row_candidates.empty:
                    continue

                mid_row = mid_row_candidates.iloc[len(mid_row_candidates) // 2]

                others_at_frame = ped_data[
                    (ped_data['frame'] == int(mid_row['frame'])) &
                    (ped_data['id'] != pid)
                ]

                is_follower = self.is_crowd_follower(mid_row, others_at_frame)

                if is_follower:
                    mode = 3
                    mode_name = 'following'
                    has_conflict = False
                    ttc_result = {
                        'delta_ttc_init': None, 'delta_ttc_final': None,
                        'ttc_ped_min': np.inf, 'ttc_veh_min': np.inf,
                        'conflict_veh_id': None,
                    }
                else:
                    # ========================================================
                    # Level 2-4: Track TTC evolution + game outcome classifier
                    # ========================================================
                    ttc_result = self.track_ttc_evolution(
                        ped_data, veh_data, pid, win_start, win_end
                    )

                    has_conflict = ttc_result['has_conflict']
                    speeds = ped_window['speed'].values

                    mode, mode_name = self.classify_game_outcome(
                        ttc_result, speeds, dt
                    )

                # Compute summary statistics
                speeds = ped_window['speed'].values
                accels = np.diff(speeds) / dt if len(speeds) > 1 else np.array([0.0])

                all_labels.append({
                    'ped_id': int(pid),
                    'frame_start': win_start,
                    'frame_end': win_end,
                    'mode': mode,
                    'mode_name': mode_name,
                    'has_conflict': has_conflict,
                    'is_follower': is_follower,
                    'mean_speed': float(np.mean(speeds)),
                    'mean_accel': float(np.mean(accels)),
                    # New: game-theoretic diagnostics
                    'delta_ttc_init': ttc_result.get('delta_ttc_init'),
                    'delta_ttc_final': ttc_result.get('delta_ttc_final'),
                    'ttc_ped_min': float(ttc_result.get('ttc_ped_min', np.inf)),
                    'ttc_veh_min': float(ttc_result.get('ttc_veh_min', np.inf)),
                    'conflict_veh_id': ttc_result.get('conflict_veh_id'),
                    'conflict_angle': ttc_result.get('mean_conflict_angle'),
                })

        return pd.DataFrame(all_labels)


# =============================================================================
# Feature extractor for GNN input (called after labeling)
# =============================================================================
class FeatureExtractor:
    """
    Extract per-agent features for GNN node input from DUT/CITR trajectory data.

    Node features (per timestep): [x, y, vx, vy, speed, accel, heading_change]
    Edge features: [distance, bearing_angle, relative_speed]
    """

    def __init__(self, fps=30.0, obs_len=150, edge_distance_threshold=10.0):
        self.fps = fps
        self.obs_len = obs_len
        self.edge_thresh = edge_distance_threshold
        self.dt = 1.0 / fps

    def extract_node_features(self, ped_data: pd.DataFrame, ped_id: int,
                              frame_start: int, frame_end: int) -> Optional[np.ndarray]:
        """
        Extract time series of node features for one pedestrian.
        Returns shape (T, 7): [x, y, vx, vy, speed, accel, heading_change]
        """
        mask = (ped_data['id'] == ped_id) & \
               (ped_data['frame'] >= frame_start) & \
               (ped_data['frame'] < frame_end)
        df = ped_data[mask].sort_values('frame')

        if len(df) < 5:
            return None

        x = df['x'].values
        y = df['y'].values
        vx = df['vx'].values
        vy = df['vy'].values
        speed = np.sqrt(vx**2 + vy**2)

        accel = np.zeros_like(speed)
        accel[1:] = np.diff(speed) / self.dt

        heading = np.arctan2(vy, vx)
        heading_change = np.zeros_like(heading)
        heading_change[1:] = np.diff(heading)
        heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi

        features = np.stack([x, y, vx, vy, speed, accel, heading_change], axis=-1)
        return features

    def build_graph_snapshot(self, ped_data: pd.DataFrame, frame: int,
                             veh_data: Optional[pd.DataFrame] = None):
        """
        Build a spatial interaction graph at a single frame.

        Returns:
            node_features: (N, 4) array [x, y, vx, vy]
            edge_index: (2, E) array of [source, target] pairs
            edge_attr: (E, 3) array [distance, bearing, relative_speed]
            agent_ids: list of agent IDs (pedestrians + vehicles)
            agent_types: list of 0/1 (0=ped, 1=veh)
        """
        peds_at_frame = ped_data[ped_data['frame'] == frame]
        agents = []

        for _, row in peds_at_frame.iterrows():
            agents.append({
                'id': int(row['id']),
                'type': 0,
                'x': row['x'], 'y': row['y'],
                'vx': row['vx'], 'vy': row['vy']
            })

        if veh_data is not None and not veh_data.empty:
            vehs_at_frame = veh_data[veh_data['frame'] == frame]
            for _, row in vehs_at_frame.iterrows():
                veh_vx = row.get('veh_vx', row.get('vel', 0) * np.cos(row.get('psi', 0)))
                veh_vy = row.get('veh_vy', row.get('vel', 0) * np.sin(row.get('psi', 0)))
                agents.append({
                    'id': int(row['id']),
                    'type': 1,
                    'x': row['x'], 'y': row['y'],
                    'vx': veh_vx, 'vy': veh_vy
                })

        if len(agents) == 0:
            return None, None, None, [], []

        N = len(agents)
        node_feats = np.array([[a['x'], a['y'], a['vx'], a['vy']] for a in agents])
        agent_ids = [a['id'] for a in agents]
        agent_types = [a['type'] for a in agents]

        src, tgt = [], []
        edge_attrs = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = node_feats[j, 0] - node_feats[i, 0]
                dy = node_feats[j, 1] - node_feats[i, 1]
                dist = math.hypot(dx, dy)

                if dist < self.edge_thresh:
                    src.append(i)
                    tgt.append(j)
                    bearing = math.atan2(dy, dx)
                    rel_speed = math.hypot(
                        node_feats[j, 2] - node_feats[i, 2],
                        node_feats[j, 3] - node_feats[i, 3]
                    )
                    edge_attrs.append([dist, bearing, rel_speed])

        edge_index = np.array([src, tgt]) if src else np.zeros((2, 0), dtype=int)
        edge_attr = np.array(edge_attrs) if edge_attrs else np.zeros((0, 3))

        return node_feats, edge_index, edge_attr, agent_ids, agent_types


# =============================================================================
# Convenience: label all scenarios in a directory
# =============================================================================
def label_all_scenarios(data_dir: str, ped_only: bool = False) -> pd.DataFrame:
    """
    Walk through a DUT/CITR data directory and label all scenarios.
    Returns a concatenated DataFrame of all labels.
    """
    import os
    labeler = PseudoLabeler()
    all_labels = []

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if ped_only:
        for f in files:
            if '_ped_' in f:
                labels = labeler.label_scenario(os.path.join(data_dir, f))
                labels['scenario'] = f
                all_labels.append(labels)
    else:
        scenario_dict = {}
        for f in files:
            key = f.split("_traj_")[0]
            scenario_dict.setdefault(key, {})
            if "_ped_" in f:
                scenario_dict[key]['ped'] = f
            elif "_veh_" in f:
                scenario_dict[key]['veh'] = f

        for key, fdict in scenario_dict.items():
            ped_f = fdict.get('ped')
            veh_f = fdict.get('veh')
            if not ped_f:
                continue

            ped_path = os.path.join(data_dir, ped_f)
            veh_path = os.path.join(data_dir, veh_f) if veh_f else None

            labels = labeler.label_scenario(ped_path, veh_path)
            labels['scenario'] = key
            all_labels.append(labels)

    if all_labels:
        return pd.concat(all_labels, ignore_index=True)
    return pd.DataFrame()
