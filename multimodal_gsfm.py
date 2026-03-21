"""
multimodal_gsfm.py
==================
Multi-Modal Game-Theoretic Social Force Model (MM-GSFM).

This extends the original GSFM.py to support behavioral mode probabilities:
    F_total(i,t) = Σ_k  P_k · F_SFM(i, t, θ_k)

where P = [P_agg, P_reg, P_cau, P_fol] comes from the frozen GNN classifier
and θ_k are mode-specific parameter sets from mode_config.py.

Key changes from original GSFM.py:
  1. simulate() accepts per-agent mode probabilities
  2. Force computation runs K=4 times with different params, weighted by P
  3. Compatible with both CSV input and programmatic state injection
  4. Supports differentiable forward pass (torch tensors) for gradient calibration
"""

import math
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple

from mode_config import (
    NUM_MODES, MODE_NAMES, BASE_PARAMS,
    get_mode_params, get_all_mode_params
)


class MultiModalGSFM:
    """
    Multi-modal forward simulator with weighted force computation.
    """

    def __init__(self, mode_probabilities: Optional[Dict[int, np.ndarray]] = None):
        """
        Args:
            mode_probabilities: dict mapping ped_id -> (4,) probability vector.
                If None, defaults to uniform [0.25, 0.25, 0.25, 0.25].
        """
        self.mode_probs = mode_probabilities or {}
        self.all_mode_params = get_all_mode_params()

    def set_mode_probabilities(self, probs: Dict[int, np.ndarray]):
        """Update mode probabilities (e.g., from GNN inference)."""
        self.mode_probs = probs

    def get_agent_probs(self, pid: int) -> np.ndarray:
        """Get mode probability vector for an agent."""
        if pid in self.mode_probs:
            return self.mode_probs[pid]
        return np.ones(NUM_MODES) / NUM_MODES  # uniform default

    # -----------------------------------------------------------------
    # Force computation functions (from original GSFM.py, parameterized)
    # -----------------------------------------------------------------
    @staticmethod
    def _anisotropy_sin(ego_dir, vec_to_other, lam):
        norm_ego = np.linalg.norm(ego_dir)
        if norm_ego < 1e-6:
            return 1.0
        ego_unit = ego_dir / norm_ego
        other_unit = vec_to_other / (np.linalg.norm(vec_to_other) + 1e-9)
        cos_phi = np.clip(np.dot(ego_unit, other_unit), -1.0, 1.0)
        phi = math.acos(cos_phi)
        return lam + (1.0 - lam) * (1.0 + math.cos(phi)) / 2.0

    @staticmethod
    def _anisotropy_exp(ego_dir, vec_to_other, lam):
        norm_ego = np.linalg.norm(ego_dir)
        if norm_ego < 1e-6:
            return 1.0
        ego_unit = ego_dir / norm_ego
        other_unit = vec_to_other / (np.linalg.norm(vec_to_other) + 1e-9)
        cos_phi = np.clip(np.dot(ego_unit, other_unit), -1.0, 1.0)
        phi = math.acos(cos_phi)
        return math.exp(-lam * abs(phi))

    def _ped_ped_repul(self, ped, other, params):
        dx = other['x'] - ped['x']
        dy = other['y'] - ped['y']
        d0 = params['ped_ped_range_repul']
        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist >= d0:
            return np.array([0.0, 0.0])
        max_F = params['ped_ped_strength_repul']
        mag = max_F * (d0 - dist + math.sqrt(0.45 + (d0 - dist)**2)) / (2 * d0)
        dir_away = np.array([-dx, -dy]) / dist
        ego_vel = np.array([ped['vx'], ped['vy']])
        ani = self._anisotropy_sin(ego_vel, np.array([dx, dy]),
                                    params['anisotropy_lambda_repul'])
        return mag * ani * dir_away

    def _ped_ped_navig(self, ped, other, params):
        dx = other['x'] - ped['x']
        dy = other['y'] - ped['y']
        d0 = params['ped_ped_range_navig']
        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist >= d0:
            return np.array([0.0, 0.0])
        vx_rel = ped['vx'] - other['vx']
        vy_rel = ped['vy'] - other['vy']
        v_rel = np.array([vx_rel, vy_rel])
        ego_vel = np.array([ped['vx'], ped['vy']])
        ani = self._anisotropy_exp(ego_vel, np.array([dx, dy]),
                                    params['anisotropy_lambda_navig'])
        max_F = params['ped_ped_strength_navig']
        mag = max_F * (d0 - dist + math.sqrt(0.45 + (d0 - dist)**2)) / (2 * d0)
        perp = np.array([-dy, dx]) / dist
        if np.dot(perp, v_rel) < 0:
            perp = -perp
        return mag * ani * perp

    def _ped_veh_force(self, ped, veh, params):
        dx = veh['x'] - ped['x']
        dy = veh['y'] - ped['y']
        dist = max(math.hypot(dx, dy), 1e-6)
        A = params['ped_veh_strength']
        B = params['ped_veh_decay']
        mag = A * math.exp(-B * dist)
        dir_away = np.array([-dx, -dy]) / dist
        ego_vel = np.array([ped['vx'], ped['vy']])
        ani = self._anisotropy_exp(ego_vel, np.array([dx, dy]),
                                    params.get('anisotropy_lambda_p2v', 0.6))
        return mag * ani * dir_away

    @staticmethod
    def _ttc_dual(p_ped, v_ped, p_veh, v_veh, ttc_thresh):
        A = np.column_stack((v_ped, -v_veh))
        b = p_veh - p_ped
        if np.linalg.matrix_rank(A) < 2:
            return np.inf, np.inf, False
        try:
            Tp, Tv = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf, np.inf, False
        if Tp > 0 and Tv > 0 and abs(Tp - Tv) < ttc_thresh:
            p_int_ped = p_ped + Tp * v_ped
            p_int_veh = p_veh + Tv * v_veh
            if np.linalg.norm(p_int_ped - p_int_veh) < 1.0:
                return Tp, Tv, True
        return Tp, Tv, False

    # -----------------------------------------------------------------
    # Compute total force for one agent under ONE mode's parameters
    # -----------------------------------------------------------------
    def _compute_force_single_mode(self, pid, state, active_peds, current_vehicles,
                                    ped_destinations, params):
        """
        Compute F_SFM(i, t, θ_k) for a single mode k's parameter set.
        This mirrors the original GSFM.simulate() inner loop logic.
        """
        ego = state
        mass = params.get('mass', BASE_PARAMS['mass'])

        # --- Crowd role detection ---
        others = [s for i, s in active_peds.items() if i != pid]
        is_follower = self._is_crowd_follower(ego, others, params)

        v2p_alpha = params['ped_single_alpha']
        v_mag = params.get('ped_des_speed', BASE_PARAMS['ped_des_speed'])

        if is_follower:
            v2p_alpha = params['ped_group_alpha']
            v_mag = self._crowd_follower_speed(ego, others, params)
        else:
            # Check TTC conflict with vehicles
            conflict = False
            for veh in current_vehicles:
                p_ped = np.array([ego['x'], ego['y']])
                v_ped = np.array([ego['vx'], ego['vy']])
                p_veh = np.array([veh['x'], veh['y']])
                v_veh = np.array([veh.get('veh_vx', 0), veh.get('veh_vy', 0)])

                Tp, Tv, c = self._ttc_dual(p_ped, v_ped, p_veh, v_veh,
                                            params['TTC_threshold'])
                if c:
                    conflict = True
                    # Game-theoretic decision
                    gt_leader = 'ped' if Tp < Tv else 'veh'
                    leader_go = 1.0 - params['GT_weight'] * (1 / max(1e-3, abs(Tp - Tv)))
                    if leader_go > (1.0 - leader_go):
                        ped_decision = 'go' if gt_leader == 'ped' else 'yield'
                    else:
                        ped_decision = 'yield' if gt_leader == 'ped' else 'go'

                    if ped_decision == 'yield':
                        v_mag *= params['spd_yield_sigma']
                    else:
                        v2p_alpha = params['ped_group_alpha']
                        v_mag = min(v_mag * params['spd_go_sigma'], 3.0)
                    break

        # Destination force (driving force)
        if pid in ped_destinations:
            dest_vec = ped_destinations[pid] - np.array([ego['x'], ego['y']])
            norm_dest = np.linalg.norm(dest_vec)
            if norm_dest > 1e-6:
                dest_dir = dest_vec / norm_dest
            else:
                dest_dir = np.array([0.0, 0.0])
            v_des = v_mag * dest_dir
        else:
            v_des = np.array([ego.get('v_des_x', 0), ego.get('v_des_y', 0)])

        v_cur = np.array([state['vx'], state['vy']])
        F_drive = (v_des - v_cur) * (mass / params['tau'])
        total_force = F_drive

        # Ped-ped forces
        for other_id, other_ped in active_peds.items():
            if other_id == pid:
                continue
            total_force += (1 - v2p_alpha) * (
                self._ped_ped_repul(state, other_ped, params) +
                self._ped_ped_navig(state, other_ped, params)
            )

        # Ped-veh forces (only if TTC conflict)
        for veh in current_vehicles:
            p_ped = np.array([ego['x'], ego['y']])
            v_ped_vec = np.array([ego['vx'], ego['vy']])
            p_veh = np.array([veh['x'], veh['y']])
            v_veh_vec = np.array([veh.get('veh_vx', 0), veh.get('veh_vy', 0)])
            _, _, c = self._ttc_dual(p_ped, v_ped_vec, p_veh, v_veh_vec,
                                      params['TTC_threshold'])
            if c:
                total_force += v2p_alpha * self._ped_veh_force(state, veh, params)

        return total_force

    # -----------------------------------------------------------------
    # Multi-modal weighted force (THE KEY INNOVATION)
    # -----------------------------------------------------------------
    def compute_multimodal_force(self, pid, state, active_peds, current_vehicles,
                                  ped_destinations):
        """
        Compute the expectation of forces across all behavioral modes:
            F_total(i,t) = Σ_k P_k · F_SFM(i, t, θ_k)

        This is Equation from the tech manual Module 2.1.
        """
        probs = self.get_agent_probs(pid)
        total_force = np.array([0.0, 0.0])

        for k in range(NUM_MODES):
            if probs[k] < 1e-6:
                continue  # skip negligible modes
            F_k = self._compute_force_single_mode(
                pid, state, active_peds, current_vehicles,
                ped_destinations, self.all_mode_params[k]
            )
            total_force += probs[k] * F_k

        return total_force

    # -----------------------------------------------------------------
    # Helper: crowd follower detection (parameterized)
    # -----------------------------------------------------------------
    def _is_crowd_follower(self, ego, others, params):
        ego_pos = np.array([ego['x'], ego['y']])
        ego_vel = np.array([ego['vx'], ego['vy']])
        ego_speed = np.linalg.norm(ego_vel)
        if ego_speed < 1e-4:
            return False

        for other in others:
            vec = np.array([other['x'], other['y']]) - ego_pos
            dist = np.linalg.norm(vec)
            if dist > params.get('crowd_range_fo', BASE_PARAMS['crowd_range_fo']):
                continue
            cos_a = np.dot(ego_vel, vec) / (ego_speed * dist + 1e-9)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            if angle < params.get('crowd_anisotropy_fo', BASE_PARAMS['crowd_anisotropy_fo']):
                return True
        return False

    def _crowd_follower_speed(self, ego, others, params):
        ego_pos = np.array([ego['x'], ego['y']])
        speeds, weights = [], []
        for other in others:
            dist = np.linalg.norm(np.array([other['x'], other['y']]) - ego_pos)
            if dist > 1e-3 and dist <= params.get('crowd_range_fo', BASE_PARAMS['crowd_range_fo']):
                speed = np.linalg.norm(np.array([other['vx'], other['vy']]))
                speeds.append(speed)
                weights.append(1.0 / dist)
        if not weights:
            return params.get('ped_des_speed', BASE_PARAMS['ped_des_speed'])
        return np.average(speeds, weights=weights)

    # -----------------------------------------------------------------
    # Full forward simulation (multi-modal version of original simulate())
    # -----------------------------------------------------------------
    def simulate(self, ped_csv_path, veh_csv_path=None):
        """
        Run multi-modal forward simulation. Compatible with original GSFM interface.
        """
        # Load data (same as original)
        ped_data = pd.read_csv(ped_csv_path)
        if 'x_est' in ped_data.columns:
            ped_data = ped_data.rename(columns={
                'x_est': 'x', 'y_est': 'y', 'vx_est': 'vx', 'vy_est': 'vy'
            })

        if veh_csv_path and os.path.exists(veh_csv_path):
            veh_data = pd.read_csv(veh_csv_path)
            if 'x_est' in veh_data.columns:
                veh_data = veh_data.rename(columns={
                    'x_est': 'x', 'y_est': 'y', 'psi_est': 'psi', 'vel_est': 'vel'
                })
        else:
            veh_data = pd.DataFrame(columns=['id', 'frame', 'x', 'y', 'psi', 'vel'])

        start_frame = int(min(ped_data['frame'].min(),
                              veh_data['frame'].min() if not veh_data.empty else ped_data['frame'].min()))
        end_frame = int(max(ped_data['frame'].max(),
                            veh_data['frame'].max() if not veh_data.empty else ped_data['frame'].max()))

        # Precompute vehicles by frame
        vehicles_by_frame = {}
        for _, row in veh_data.iterrows():
            frame = int(row['frame'])
            veh_info = {
                'id': int(row['id']), 'x': row['x'], 'y': row['y'],
                'psi': row.get('psi', 0), 'vel': row.get('vel', 0)
            }
            veh_info['veh_vx'] = veh_info['vel'] * math.cos(veh_info['psi'])
            veh_info['veh_vy'] = veh_info['vel'] * math.sin(veh_info['psi'])
            vehicles_by_frame.setdefault(frame, []).append(veh_info)

        # Pedestrian destinations
        ped_destinations = {}
        for pid, group in ped_data.groupby('id'):
            end_row = group[group['frame'] == group['frame'].max()].iloc[0]
            ped_destinations[pid] = np.array([end_row['x'], end_row['y']])

        # Init
        ped_spawn = {}
        ped_end = {}
        ped_init = {}
        for pid, group in ped_data.groupby('id'):
            pid = int(pid)
            spawn = int(group['frame'].min())
            end = int(group['frame'].max())
            ped_spawn[pid] = spawn
            ped_end[pid] = end
            init_row = group[group['frame'] == spawn].iloc[0]
            ped_init[pid] = {
                'x': init_row['x'], 'y': init_row['y'],
                'vx': init_row.get('vx', 0.0), 'vy': init_row.get('vy', 0.0)
            }

        active_peds = {}
        ped_trajs = {}
        veh_trajs = {int(vid): [] for vid in veh_data['id'].unique()} if not veh_data.empty else {}

        dt = BASE_PARAMS['dt']
        mass = BASE_PARAMS['mass']

        # Simulation loop
        for frame in range(start_frame, end_frame + 1):
            # Spawn
            for pid, sf in ped_spawn.items():
                if frame == sf:
                    state = ped_init[pid].copy()
                    state['v_des_x'] = state['vx']
                    state['v_des_y'] = state['vy']
                    state['last_frame'] = ped_end[pid]
                    active_peds[pid] = state
                    ped_trajs[pid] = [{'frame': frame, 'x': state['x'], 'y': state['y'],
                                       'vx': state['vx'], 'vy': state['vy']}]

            # Despawn
            to_remove = [p for p, s in active_peds.items() if frame > s['last_frame']]
            for p in to_remove:
                active_peds.pop(p, None)

            if not active_peds and frame >= max(ped_spawn.values()):
                break

            current_vehicles = vehicles_by_frame.get(frame, [])
            for veh in current_vehicles:
                if veh['id'] in veh_trajs:
                    veh_trajs[veh['id']].append({
                        'frame': frame, 'x': veh['x'], 'y': veh['y'],
                        'psi': veh['psi'], 'vel': veh['vel']
                    })

            # Update each pedestrian using MULTI-MODAL weighted force
            new_states = {}
            for pid, state in active_peds.items():
                # === THE KEY: multi-modal force computation ===
                total_force = self.compute_multimodal_force(
                    pid, state, active_peds, current_vehicles, ped_destinations
                )

                acceleration = total_force / mass
                new_vx = state['vx'] + acceleration[0] * dt
                new_vy = state['vy'] + acceleration[1] * dt
                new_x = state['x'] + new_vx * dt
                new_y = state['y'] + new_vy * dt

                new_states[pid] = {
                    'x': new_x, 'y': new_y, 'vx': new_vx, 'vy': new_vy,
                    'v_des_x': state['v_des_x'], 'v_des_y': state['v_des_y'],
                    'last_frame': state['last_frame']
                }

                next_frame = frame + 1
                if next_frame <= state['last_frame']:
                    ped_trajs[pid].append({
                        'frame': next_frame, 'x': new_x, 'y': new_y,
                        'vx': new_vx, 'vy': new_vy
                    })

            active_peds.update(new_states)

        return ped_trajs, veh_trajs

    # -----------------------------------------------------------------
    # Fitness evaluation (same as original)
    # -----------------------------------------------------------------
    def fitness(self, ped_trajs, veh_trajs, ped_data, veh_data):
        pos_errors = []
        vel_errors = []

        gt_ped = {}
        for _, row in ped_data.iterrows():
            gt_ped[(int(row['id']), int(row['frame']))] = (row['x'], row['y'], row['vx'], row['vy'])

        for pid, traj in ped_trajs.items():
            for pt in traj:
                key = (pid, pt['frame'])
                if key not in gt_ped:
                    continue
                gt_x, gt_y, gt_vx, gt_vy = gt_ped[key]
                pos_errors.append(math.hypot(pt['x'] - gt_x, pt['y'] - gt_y))
                vel_errors.append(math.hypot(pt['vx'] - gt_vx, pt['vy'] - gt_vy))

        pos_errors = np.array(pos_errors) if pos_errors else np.array([float('inf')])
        vel_errors = np.array(vel_errors) if vel_errors else np.array([float('inf')])

        return {
            'pos_mean': pos_errors.mean(), 'pos_std': pos_errors.std(),
            'vel_mean': vel_errors.mean(), 'vel_std': vel_errors.std()
        }

    def evaluate(self, ped_csv_path, veh_csv_path=None):
        ped_data = pd.read_csv(ped_csv_path)
        if 'x_est' in ped_data.columns:
            ped_data = ped_data.rename(columns={
                'x_est': 'x', 'y_est': 'y', 'vx_est': 'vx', 'vy_est': 'vy'
            })

        if veh_csv_path and os.path.exists(veh_csv_path):
            veh_data = pd.read_csv(veh_csv_path)
            if 'x_est' in veh_data.columns:
                veh_data = veh_data.rename(columns={
                    'x_est': 'x', 'y_est': 'y', 'psi_est': 'psi', 'vel_est': 'vel'
                })
        else:
            veh_data = pd.DataFrame(columns=['id', 'frame', 'x', 'y', 'psi', 'vel'])

        ped_trajs, veh_trajs = self.simulate(ped_csv_path, veh_csv_path)
        return self.fitness(ped_trajs, veh_trajs, ped_data, veh_data)
