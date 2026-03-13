import math
import pandas as pd
import numpy as np
import os

class GSFM:
    """
    A forward simulation of the Game-Theoretic Social Force Model (GSFM) for pedestrian dynamics.
    This model includes pedestrian-pedestrian and pedestrian-vehicle interactions with anisotropic sensitivity.
    """
    def __init__(self, params=None):
        """
        Initialize the GSFM model with given parameters or defaults.
        Parameters:
            params (dict): Optional dictionary of model parameters. Keys can include:
                - 'dt': time step in seconds (frame duration, default ~0.0333s for 30 FPS)
                - 'tau': relaxation time for reaching desired velocity (s)
                - 'ped_ped_strength': base magnitude of pedestrian-pedestrian repulsive force
                - 'ped_ped_range': effective range (m) of pedestrian-pedestrian interaction (force goes to 0 at this distance)
                - 'ped_veh_strength': base magnitude of pedestrian-vehicle repulsive force
                - 'ped_veh_decay': decay rate (1/m) for pedestrian-vehicle interaction (exponential decay)
                - 'anisotropy_lambda_repul': lambda parameter (0 to 1) for sinusoidal anisotropy function in repulsive force
                - 'anisotropy_lambda_navig': lambda parameter (0 to 1) for sinusoidal anisotropy function in navigation force
                - 'mass': mass of a pedestrian (kg, can be normalized to 1 for simplicity)
        """
        # Default parameters (tunable for calibration)
        default_params = {
            'dt': 0.0333,  # time step (~30 Hz by default)            # time step (~30 Hz by default)
            'tau': 1.63,
            'ped_ped_range_repul': 2.0, 
            'ped_ped_range_navig': 0.79,
            'anisotropy_lambda_repul': 0.1, 
            'anisotropy_lambda_navig': 0.28, 
            'ped_single_alpha': 0.72, 
            'ped_group_alpha': 0.63, 
            'ped_veh_strength': 32.51, 
            'ped_veh_decay': 0.32, 
            'anisotropy_lambda_p2v': 0.6, 
            'ped_ped_strength_repul': 6.3, 
            'ped_ped_strength_navig': 4.45, 
            'ped_des_speed': 1.44, 
            'destination_sigma': 1.63,
            'des_strength': 3.1, 
            'mass': 1,  # mass of pedestrian agent
            'crowd_anisotropy_fo': 60,  # angle threshold (deg) to determine field of view for follower detection
            'spd_go_sigma':1.5,
            'crowd_range_fo': 2.68, 
            'TTC_threshold': 2.60, 
            'spd_yield_sigma': 0.68,
            'GT_weight': 1.0,
        }
        # Override defaults with any user-provided parameters
        if params:
            default_params.update(params)
        self.params = default_params
    
    def load_data(self, ped_csv_path, veh_csv_path=None):
        ped_data = pd.read_csv(ped_csv_path)
        if veh_csv_path and os.path.exists(veh_csv_path):
            veh_data = pd.read_csv(veh_csv_path)
        else:
            veh_data = pd.DataFrame(columns=['id', 'frame', 'x', 'y', 'psi', 'vel'])
        
        if 'x_est' in ped_data.columns:
            ped_data = ped_data.rename(columns={
                'x_est': 'x', 'y_est': 'y', 'vx_est': 'vx', 'vy_est': 'vy'
            })
        if not veh_data.empty and 'x_est' in veh_data.columns:
            veh_data = veh_data.rename(columns={
                'x_est': 'x', 'y_est': 'y', 'psi_est': 'psi', 'vel_est': 'vel'
            })
        return ped_data, veh_data
    
    def anisotropy_sin_factor(self, ego_direction, vector_to_other, lam):
        """
        Compute the anisotropy weight A(φ) for an interaction, given the ego pedestrian's direction and the vector to the other agent.
        Uses sinusoidal anisotropy: A(φ) = λ + (1 - λ) * (1 + cos|φ|) / 2,
        where φ is the angle between ego's heading direction and the line-of-sight to the other agent.
        """
        # If pedestrian is nearly stationary (very low speed), assume no preferred direction (treat as isotropic)
        norm_ego = np.linalg.norm(ego_direction)
        if norm_ego < 1e-6:
            return 1.0  # no directional bias if not moving
        # Unit vectors for ego direction and direction to other
        ego_dir_unit = ego_direction / norm_ego
        other_dir_unit = vector_to_other / (np.linalg.norm(vector_to_other) + 1e-9)
        # Compute angle via dot product
        cos_phi = np.clip(np.dot(ego_dir_unit, other_dir_unit), -1.0, 1.0)
        phi = math.acos(cos_phi)  # angle between [0, π]
        # Sinusoidal anisotropy: smoothly attenuates influence as angle increases
        return lam + (1.0 - lam) * (1.0 + math.cos(phi)) / 2.0
    
    
    def anisotropy_exp_factor(self, ego_direction, vector_to_other, lam):
        """
        Compute the anisotropy weight A(φ) for an interaction, given the ego pedestrian's direction and the vector to the other agent.
        Uses sinusoidal anisotropy: A(φ) = λ + (1 - λ) * (1 + cos|φ|) / 2,
        where φ is the angle between ego's heading direction and the line-of-sight to the other agent.
        """
        # If pedestrian is nearly stationary (very low speed), assume no preferred direction (treat as isotropic)
        norm_ego = np.linalg.norm(ego_direction)
        if norm_ego < 1e-6:
            return 1.0  # no directional bias if not moving
        # Unit vectors for ego direction and direction to other
        ego_dir_unit = ego_direction / norm_ego
        other_dir_unit = vector_to_other / (np.linalg.norm(vector_to_other) + 1e-9)
        # Compute angle via dot product
        cos_phi = np.clip(np.dot(ego_dir_unit, other_dir_unit), -1.0, 1.0)
        phi = math.acos(cos_phi)  # angle between [0, π]
        # Sinusoidal anisotropy: smoothly attenuates influence as angle increases
        return math.exp(-lam * abs(phi))
    
    def compute_ped_ped_repul(self, ped_state, other_state):
        """
        Compute repulsive force (vector) on one pedestrian due to another pedestrian.
        Uses linear decay with smoothness
        Applies sinusoidal anisotropy to reduce force if other is not in front of ego.
        """
        # Relative position from ego ped to other ped
        dx = other_state['x'] - ped_state['x']
        dy = other_state['y'] - ped_state['y']
        d0 = self.params['ped_ped_range_repul']

        # Check if within interaction range
        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist >= d0:
            # Pedestrians are at virtually the same position (avoid division by zero)
            return np.array([0.0, 0.0])
            
        # Compute base force magnitude with linear decay
        max_F = self.params['ped_ped_strength_repul']
        mag = max_F * (d0 - dist + math.sqrt(0.45 + (d0-dist)**2)) / (2*d0)  # linear decrease to 0 at d = d0

        # Direction of force: from other to ego (repulsion pushes ego away from other)
        dir_away = np.array([-dx, -dy]) / dist  # unit vector pointing from other to ego

        # Anisotropy factor based on ego's velocity direction
        ego_vel = np.array([ped_state['vx'], ped_state['vy']])
        ani = self.anisotropy_sin_factor(ego_vel, np.array([dx, dy]), self.params['anisotropy_lambda_repul'])

        # Final force vector
        return mag * ani * dir_away
    
    def compute_ped_ped_navig(self, ped_state, other_state):
        """
        Compute navigation force (vector) on one pedestrian due to another pedestrian.
        The force acts perpendicular to the direction between ego and the other pedestrian,
        encouraging deviation to avoid head-on collision paths.
        """
        # Relative position from ego to other
        dx = other_state['x'] - ped_state['x']
        dy = other_state['y'] - ped_state['y']
        d0 = self.params['ped_ped_range_navig']

        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist >= d0:
            return np.array([0.0, 0.0])

        # Relative velocity: ego - other
        vx_rel = ped_state['vx'] - other_state['vx']
        vy_rel = ped_state['vy'] - other_state['vy']
        v_rel = np.array([vx_rel, vy_rel])
        r_vec = np.array([dx, dy])

        
        # Anisotropy effect using exponential decay
        ego_vel = np.array([ped_state['vx'], ped_state['vy']])
        ani = self.anisotropy_exp_factor(ego_vel, np.array([dx, dy]), self.params['anisotropy_lambda_navig'])

        # Navigation force magnitude 
        max_F = self.params['ped_ped_strength_navig']
        mag = max_F * (d0 - dist + math.sqrt(0.45 + (d0-dist)**2)) / (2*d0)  # linear decrease to 0 at d = d0

        # Perpendicular unit vector to r_vec (clockwise)
        perp_vec = np.array([-dy, dx]) / dist

        # Ensure perp_vec is on same side as v_rel
        if np.dot(perp_vec, v_rel) < 0:
            perp_vec = -perp_vec

        # Final force vector
        return mag * ani * perp_vec

    
    def compute_ped_veh_force(self, ped_state, veh_state):
        """
        Compute repulsive force (vector) on a pedestrian due to a vehicle.
        Uses exponential decay: force magnitude = ped_veh_strength * exp(-B * d).
        Also applies sinusoidal anisotropy (pedestrian is less sensitive to vehicles behind them).
        """
        # Relative position from pedestrian to vehicle
        dx = veh_state['x'] - ped_state['x']
        dy = veh_state['y'] - ped_state['y']
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            # Pedestrian is extremely close to or at the vehicle's position (potential collision)
            dist = 1e-6  # avoid singularity by using a small distance
        # Base force magnitude with exponential decay
        A = self.params['ped_veh_strength']
        B = self.params['ped_veh_decay']
        mag = A * math.exp(-B * dist)
        # Direction of force: from vehicle to pedestrian (pushes pedestrian away from vehicle)
        dir_away = np.array([-dx, -dy]) / dist  # unit vector from vehicle toward ped
        # Anisotropy factor for pedestrian w.r.t. vehicle
        ego_vel = np.array([ped_state['vx'], ped_state['vy']])
        ani = self.anisotropy_exp_factor(ego_vel, np.array([dx, dy]),self.params['anisotropy_lambda_p2v'])
        return mag * ani * dir_away
    
    def compute_destination_force(self, ped_id, ped_state, des_spd):
        """
        Compute destination force that guides the pedestrian toward a predefined goal.
        This simplified version assumes the pedestrian always tries to go toward the goal
        with fixed feedback gain, without dynamic suppression from vehicles.
        """
        if ped_id not in self.ped_destinations:
            return np.array([0.0, 0.0])

        # Current position and velocity
        pos = np.array([ped_state['x'], ped_state['y']])
        dest = np.array(self.ped_destinations[ped_id])
        direction = dest - pos
        dist2 = np.dot(direction, direction)

        if dist2 < 1e-6:
            v_des = np.array([0.0, 0.0])
        else:
            v_des = des_spd * direction / np.sqrt(dist2 + self.params['destination_sigma']**2)

        v_cur = np.array([ped_state['vx'], ped_state['vy']])

        # Classic destination force
        return self.params['des_strength'] * (v_des - v_cur)

    def is_crowd_follower(self, ego, others):
        ego_pos = np.array([ego['x'], ego['y']])
        ego_vel = np.array([ego['vx'], ego['vy']])
        leaders = []
        for other in others:
            if other == ego:
                continue
            vec_to_other = np.array([other['x'], other['y']]) - ego_pos
            dist = np.linalg.norm(vec_to_other)
            if dist > self.params['crowd_range_fo']:
                continue
            angle = self.calculate_angle(ego_vel, vec_to_other)
            if abs(angle) < self.params['crowd_anisotropy_fo']:
                leaders.append(other)
        return leaders

    def compute_crowd_follower_speed(self, ego, leaders):
        ego_pos = np.array([ego['x'], ego['y']])
        speeds = []
        weights = []
        for leader in leaders:
            dist = np.linalg.norm(np.array([leader['x'], leader['y']]) - ego_pos)
            speed = np.linalg.norm(np.array([leader['vx'], leader['vy']]))
            if dist > 1e-3:
                speeds.append(speed)
                weights.append(1.0 / dist)
        if not weights:
            return self.params['ped_des_speed']
        return np.average(speeds, weights=weights)

    def calculate_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 180.0
        cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def ttc_dual(self, p_ped, v_ped, p_veh, v_veh, dist_tol=1.0):
        """
        计算行人‐车辆双时间 TTC。
        返回: TTC_scalar, Tp, Tv, risk_flag
        risk_flag 为 True 的条件:
            Tp > 0, Tv > 0 且 |Tp - Tv| < self.params['TTC_threshold']
        """
        # ----------------------------------------------------------------
        # 构造线性方程组  p_ped + Tp*v_ped = p_veh + Tv*v_veh
        A = np.column_stack((v_ped, -v_veh))
        b = p_veh - p_ped

        # 若秩不足 2，则说明两条轨迹平行或重合，单独处理
        if np.linalg.matrix_rank(A) < 2:
            return np.inf, np.inf, False

        # 求 Tp, Tv
        Tp, Tv = np.linalg.solve(A, b)

        # ----------------------------------------------------------------
        # 判定风险: 时间同步条件
        if Tp > 0 and Tv > 0 and abs(Tp - Tv) < self.params['TTC_threshold']: #时间几乎同步
            p_int_ped = p_ped + Tp * v_ped
            p_int_veh = p_veh + Tv * v_veh
            if np.linalg.norm(p_int_ped - p_int_veh) < dist_tol: #空间相互逼近
                return Tp, Tv, True
        return Tp, Tv, False
    
    def compute_conflict_loss(self, dij, delta_ttc):
        Ac = self.params.get('conflict_loss_A', 1.0)
        Dcrit = self.params.get('conflict_loss_Dcrit', 2.0)
        Bc = self.params.get('conflict_loss_B', 1.0)
        return Ac * max(Dcrit, Dcrit - dij) * math.exp(-Bc * abs(delta_ttc))

    def compute_waiting_loss(self, d_tail, v_tail):
        tau = self.params['tau']
        return d_tail / (v_tail + tau)
    
    def sigmoid(self, z, k=1.0):
        return 1 / (1 + math.exp(-k * z))

    def compute_utilities(self, L_conf, L_wait, delta_ttc):
        s = self.sigmoid(delta_ttc, k=self.params.get('GT_weight', 1.0))
        u_go = -L_conf * (1 - s)
        u_yield = -L_wait * s
        return u_go, u_yield

    def simulate(self, ped_csv_path, veh_csv_path):
        """
        Run the forward simulation for all pedestrians across time.
        Initializes pedestrians and vehicles from input files, then iteratively updates pedestrian states.
        """
        # 1. Load trajectory data for pedestrians and vehicles
        ped_data, veh_data = self.load_data(ped_csv_path, veh_csv_path)

        # Determine simulation start and end frames from data
        start_frame = int(min(ped_data['frame'].min(), veh_data['frame'].min()))
        end_frame   = int(max(ped_data['frame'].max(), veh_data['frame'].max()))

        # Precompute vehicle positions by frame for quick lookup
        vehicles_by_frame = {}
        for _, row in veh_data.iterrows():
            frame = int(row['frame'])
            veh_info = {
                'id':  int(row['id']),
                'x':   row['x'],
                'y':   row['y'],
                # Orientation and speed can be stored for future use (e.g., if modeling vehicle movement)
                'psi': row.get('psi', None),
                'vel': row.get('vel', None)
            }
            vehicles_by_frame.setdefault(frame, []).append(veh_info)

        # Assign destination as the last known position from trajectory
        self.ped_destinations = {}
        for ped_id, group in ped_data.groupby('id'):
            end_row = group[group['frame'] == group['frame'].max()].iloc[0]
            self.ped_destinations[ped_id] = np.array([end_row['x'], end_row['y']])


        # Initialize vehicle output container
        veh_trajs = {int(vid): [] for vid in veh_data['id'].unique()}

        # Prepare dictionaries for pedestrian spawn/despawn frames and initial states
        ped_spawn_frame = {}
        ped_end_frame   = {}
        ped_init_state  = {}
        for ped_id, group in ped_data.groupby('id'):
            ped_id = int(ped_id)
            spawn = int(group['frame'].min())
            end   = int(group['frame'].max())
            ped_spawn_frame[ped_id] = spawn
            ped_end_frame[ped_id]   = end
            # Initial state at spawn frame
            init_row = group[group['frame'] == spawn].iloc[0]
            ped_init_state[ped_id] = {
                'x':  init_row['x'],
                'y':  init_row['y'],
                'vx': init_row.get('vx', 0.0),
                'vy': init_row.get('vy', 0.0)
            }

        # Containers for active pedestrians and output trajectories
        active_peds = {}       # dict of ped_id -> state
        ped_trajs   = {}       # dict of ped_id -> list of state dicts over time
        dt = self.params['dt']
        mass = self.params['mass']

        # 2. Simulation loop through each frame
        for frame in range(start_frame, end_frame + 1):
            # (a) Spawn new pedestrians at their start frame
            for ped_id, spawn_f in ped_spawn_frame.items():
                if frame == spawn_f:
                    # Initialize state for new pedestrian
                    state = ped_init_state[ped_id].copy()
                    # Use initial velocity as desired velocity (could be adjusted if needed)
                    state['v_des_x'] = state['vx']
                    state['v_des_y'] = state['vy']
                    state['last_frame'] = ped_end_frame[ped_id]
                    active_peds[ped_id] = state
                    # Initialize trajectory list and record initial position
                    ped_trajs[ped_id] = []
                    ped_trajs[ped_id].append({
                        'frame': frame, 'x': state['x'], 'y': state['y'],
                        'vx':    state['vx'], 'vy': state['vy']
                    })

            # (b) Remove pedestrians that have reached their end frame (they leave the scene)
            to_remove = [pid for pid, st in active_peds.items() if frame > st['last_frame']]
            for pid in to_remove:
                active_peds.pop(pid, None)

            # If no active pedestrians and no new ones will spawn later, we can stop early
            if not active_peds and frame >= max(ped_spawn_frame.values()):
                break

            # (c) Get current vehicles (if any) at this frame and log them
            current_vehicles = vehicles_by_frame.get(frame, [])
            for veh in current_vehicles:
                veh_trajs[veh['id']].append({
                    'frame': frame,
                    'x':     veh['x'],
                    'y':     veh['y'],
                    'psi':   veh['psi'],
                    'vel':   veh['vel']
                })

            for veh in current_vehicles:
                if 'veh_vx' not in veh:
                    veh['veh_vx'] = veh['vel'] * math.cos(veh['psi'])
                    veh['veh_vy'] = veh['vel'] * math.sin(veh['psi'])

            # (d) Compute forces and update each active pedestrian
            new_states = {}  # to store updated state for each ped after this time step
           
            for pid, state in active_peds.items():
                
                # Game Theoretic Controller Layer
                ego = state
                others = [s for i, s in active_peds.items() if i != pid]
                
                leaders = self.is_crowd_follower(ego, others)
                v2p_alpha = self.params['ped_single_alpha'] #assume pedestrian soley interacte with vehicle
                                
                if leaders: # ego is a crowd follower
                    v2p_alpha = self.params['ped_group_alpha']
                    v_mag = self.compute_crowd_follower_speed(ego, leaders) #pedestrain interacte wtih vehicle within a crowd
                    dest_vec = self.ped_destinations[pid] - np.array([ego['x'], ego['y']])
                    norm_dest = np.linalg.norm(dest_vec)
                    if norm_dest > 1e-6:
                        dest_dir = dest_vec / norm_dest
                    else:
                        dest_dir = np.array([0.0, 0.0]) 
                    ego['v_des_x'] = v_mag * dest_dir[0]
                    ego['v_des_y'] = v_mag * dest_dir[1]

                else: #ego is a crowd leader
                    conflict = False
                    v_mag = self.params['ped_des_speed']
                                                            
                    for veh in current_vehicles:
                        p_ped = np.array([ego['x'], ego['y']])
                        v_ped = np.array([ego['vx'], ego['vy']])
                        p_veh = np.array([veh['x'], veh['y']])
                        v_veh = np.array([veh['veh_vx'], veh['veh_vy']])

                        # Estimate TTC using relative motion: solving A t = b where A = [v1, -v2], b = p2 - p1
                        TTC_ped, TTC_veh, conflict = self.ttc_dual(p_ped, v_ped, p_veh, v_veh, dist_tol=1.0)

                    if conflict:
                        GT_leader = 'ped' if TTC_ped < TTC_veh else 'veh'
                        
                        Leader_go = 1.0 - self.params['GT_weight'] * (1 / max(1e-3,abs(TTC_ped - TTC_veh))) 
                        Leader_yield = 1.0 - Leader_go

                        if Leader_go > Leader_yield:
                            ped_decision = 'go' if GT_leader == 'ped' else 'yield'
                        else:
                            ped_decision = 'yield' if GT_leader == 'ped' else 'go'

                        if ped_decision == 'yield':
                            v_mag = v_mag * self.params['spd_yield_sigma'] #desired speed fall to a lower level to yield to veh
                        
                        else:
                            v2p_alpha = self.params['ped_group_alpha'] # if continue, then the influence weight of veh fall to a lower level
                            v_mag = min(v_mag * self.params['spd_go_sigma'],3)
                    
                    dest_vec = self.ped_destinations[pid] - np.array([ego['x'], ego['y']])
                    norm_dest = np.linalg.norm(dest_vec)
                    if norm_dest > 1e-6:
                        dest_dir = dest_vec / norm_dest
                    else:
                        dest_dir = np.array([0.0, 0.0])
                    ego['v_des_x'] = v_mag * dest_dir[0]
                    ego['v_des_y'] = v_mag * dest_dir[1]
                
                
                # Compute driving force: steer velocity toward desired velocity
                v_des = np.array([ego['v_des_x'], ego['v_des_y']])
                v_cur = np.array([state['vx'], state['vy']])
                
                # Driving force F_drive = (v_des - v_cur) * (mass/τ). (Here mass=1 simplifies to (v_des - v_cur)/τ)
                F_drive = (v_des - v_cur) * (mass / self.params['tau'])
                total_force = F_drive  # start with driving force
                """
                # Add destination force
                total_force += self.compute_destination_force(pid, state, des_spd = self.params['ped_des_speed'])
                """

                # Add forces from other pedestrians
                for other_id, other_ped in active_peds.items():
                    if other_id == pid:
                        continue
                    total_force += (1 - v2p_alpha) * (self.compute_ped_ped_repul(state, other_ped) + self.compute_ped_ped_navig(state, other_ped))

                # Add forces from vehicles                           
                for veh in current_vehicles:
                    # postion and velocity
                    p_ped = np.array([ego['x'], ego['y']])
                    v_ped = np.array([ego['vx'], ego['vy']])
                    p_veh = np.array([veh['x'], veh['y']])
                    v_veh = np.array([veh['veh_vx'], veh['veh_vy']])
                    TTC_ped, TTC_veh, conflict = self.ttc_dual(p_ped, v_ped, p_veh, v_veh, dist_tol=1.0)
                    if conflict:
                        total_force += v2p_alpha * self.compute_ped_veh_force(state, veh)

                # Compute acceleration (a = F / m)
                acceleration = total_force / mass

                # Euler integration: update velocity and position
                new_vx = state['vx'] + acceleration[0] * dt
                new_vy = state['vy'] + acceleration[1] * dt
                new_x  = state['x']  + new_vx * dt
                new_y  = state['y']  + new_vy * dt

                # Save updated state (carry forward desired velocity and last_frame)
                new_states[pid] = {
                    'x':       new_x, 'y': new_y,
                    'vx':      new_vx, 'vy': new_vy,
                    'v_des_x': state['v_des_x'], 'v_des_y': state['v_des_y'],
                    'last_frame': state['last_frame']
                }

                # Record this new state in the pedestrian trajectory output
                next_frame = frame + 1
                if next_frame <= state['last_frame']:
                    ped_trajs[pid].append({
                        'frame': next_frame, 'x': new_x, 'y': new_y,
                        'vx':    new_vx, 'vy': new_vy
                    })

            # (e) Update active pedestrians for the next iteration
            active_peds.update(new_states)

        return ped_trajs, veh_trajs
    
    def fitness(self, ped_trajs, veh_trajs, ped_data, veh_data):
        """
        Compute fitness metrics (mean and std of position and velocity errors)
        between simulated trajectories and ground truth.
        """
        pos_errors = []
        vel_errors = []
        # --- pedestrians ---
        # build a lookup dict for ground truth by (ped_id, frame)
        gt_ped = {}
        for _, row in ped_data.iterrows():
            gt_ped[(int(row['id']), int(row['frame']))] = (row['x'], row['y'], row['vx'], row['vy'])
        
        for pid, traj in ped_trajs.items():
            for pt in traj:
                key = (pid, pt['frame'])
                if key not in gt_ped:
                    continue
                gt_x, gt_y, gt_vx, gt_vy = gt_ped[key]
                # position error
                dx = pt['x'] - gt_x
                dy = pt['y'] - gt_y
                pos_errors.append(math.hypot(dx, dy))
                # velocity error (vector norm)
                dvx = pt['vx'] - gt_vx
                dvy = pt['vy'] - gt_vy
                vel_errors.append(math.hypot(dvx, dvy))
        
        # --- vehicles ---
        gt_veh = {}
        for _, row in veh_data.iterrows():
            gt_veh[(int(row['id']), int(row['frame']))] = (row['x'], row['y'], row['vel'])
        
        for vid, traj in veh_trajs.items():
            for vt in traj:
                key = (vid, vt['frame'])
                if key not in gt_veh:
                    continue
                gt_x, gt_y, gt_vel = gt_veh[key]
                # position error
                dx = vt['x'] - gt_x
                dy = vt['y'] - gt_y
                pos_errors.append(math.hypot(dx, dy))
                # speed error (scalar)
                vel_errors.append(abs(vt['vel'] - gt_vel))
        
        pos_errors = np.array(pos_errors)
        vel_errors = np.array(vel_errors)
        return {
            'pos_mean': pos_errors.mean(),
            'pos_std':  pos_errors.std(),
            'vel_mean': vel_errors.mean(),
            'vel_std':  vel_errors.std()
        }


    def evaluate(self, ped_csv_path, veh_csv_path = None):
        """
        Run simulate on the two CSVs, then compute and print fitness.
        """
        # load ground truth
        ped_truth, veh_truth = self.load_data(ped_csv_path, veh_csv_path)
        # run simulation
        ped_smlt, veh_smlt = self.simulate(ped_csv_path, veh_csv_path)
        # compute fitness
        metrics = self.fitness(ped_smlt, veh_smlt, ped_truth, veh_truth)
        #print(f"Position error - mean: {metrics['pos_mean']:.3f}, std: {metrics['pos_std']:.3f}")
        #print(f"Velocity error - mean: {metrics['vel_mean']:.3f}, std: {metrics['vel_std']:.3f}")
        return metrics
