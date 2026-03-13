import os
import json
import math
import numpy as np

# 设置距离、速度和方向阈值
INTERACTION_THRESHOLD = 1.0  # 距离阈值（米）
SPEED_THRESHOLD = 0.5  # 相对速度阈值（m/s)
ACCELERATION_THRESHOLD = 0.3
PEDES_SPEED_THRESHOLD = 1.5
TURN_THRESHOLD = 15 # turning angle threshold
ANGLE_THRESHOLD = 45
TTC_MAX = 0.5
TTC_RANGE = [0,TTC_MAX]        #TTC threshold
CV_THRESHOLD = 0.3       #coefficient_of_variation threshold

# Function to check if JSON file is empty or contains no meaningful data
def is_valid_json(file_path):
    try:
        with open(file_path, "r") as f:
            json_content = json.load(f)
        
        # Check if all relevant fields are empty
        if not any(json_content.get(key) for key in [
            "past_vehicle_tracks", "past_pedestrian_tracks",
            "future_vehicle_tracks", "future_pedestrian_tracks"
        ]):
            return False  # No valid data in the JSON
        
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False  # Skip if file is unreadable or malformed
    
# Calculates Time to Collision (TTC)
def calculate_TTC(p1, p2, v1, v2):
        # Construct matrix A and vector b:
    #   [ v1x  -v2x ] [ t1 ] = [ p2x - p1x ]
    #   [ v1y  -v2y ] [ t2 ]   [ p2y - p1y ]
    A = np.array([[v1[0], -v2[0]],
                  [v1[1], -v2[1]]], dtype=float)
    b = np.array([p2[0] - p1[0],
                  p2[1] - p1[1]], dtype=float)

    # Check if A is invertible (determinant should not be 0)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return (None, None)

    # Solve for t1, t2 using numpy.linalg.solve
    t1, t2 = np.linalg.solve(A, b)
    return (t1, t2)

# 计算欧几里得距离
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 计算相对速度
def calculate_relative_speed(vehicle_velocity, pedestrian_velocity):
    vx1, vy1 = vehicle_velocity
    vx2, vy2 = pedestrian_velocity
    return math.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2) #相对速度计算公式有误

# Calculate the angle change by the cosine
def calculate_angle(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude_v2 = math.sqrt(sum(b ** 2 for b in v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))

# 动态调整距离阈值（基于车辆速度）
def dynamic_threshold(speed):
    return INTERACTION_THRESHOLD + 0.5 * speed

def calculate_coefficient_of_variation(speed_list):
    """
   Compute the coefficient of variation (CV) for a given list of speed magnitudes
    """
    if not speed_list:
        return 0.0
    mean_speed = np.mean(speed_list)
    if abs(mean_speed) < 1e-6:
        return 0.0
    std_speed = np.std(speed_list)
    return std_speed / mean_speed

def get_position_and_velocity(path, frame):
    for f, x, y, vx, vy in path:
        if f == frame:
            return (x, y), (vx, vy)
    return None, None

# 判断距离和角度
def check_dis_angle(vehicle_position,vehicle_velocity,pedestrian_position, pedestrian_velocity):

    # 计算距离
    distance = calculate_distance(*vehicle_position, *pedestrian_position)

    # 动态调整距离阈值
    adjusted_distance_threshold = dynamic_threshold(math.sqrt(vehicle_velocity[0]**2 + vehicle_velocity[1]**2))


    # 计算方向夹角余弦值
    angle_change = calculate_angle(vehicle_velocity, pedestrian_velocity)

    # 判断是否交互
    if (distance <= adjusted_distance_threshold and 
        angle_change >= ANGLE_THRESHOLD):
        return True
    
    return False

# 提取轨迹数据
def extract_positions(track_data, target_dict):
    if not track_data:  # 检查 track_data 是否为空
        return
    
    for time_key, tracks in track_data[0].items():
        for track in tracks:
            track_id = track['track_id']
            position = track['position']
            velocity = track['linear_velocity']
            if track_id not in target_dict:
                target_dict[track_id] = []
            target_dict[track_id].append((time_key, position['x'], position['y'], velocity['x'], velocity['y']))

# 遍历文件夹并处理所有 JSON 文件
def process_all_files(root_folder_path, output_folder):
    interaction_files = []
    total_files = 0
    all_files = []
    total_folders = 0

    for root, dirs, files in os.walk(root_folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                all_files.append(os.path.join(root, file_name))
                total_files += 1
        if dirs:
            total_folders += 1

    for folder_idx, (root, dirs, files) in enumerate(os.walk(root_folder_path), 1):
        print(f"\nProcessing folder {folder_idx}/{total_folders}: {root}")
        folder_files = [os.path.join(root, file_name) for file_name in files if file_name.endswith('.json')]
        total_files_in_folder = len(folder_files)

        for file_idx, file_path in enumerate(folder_files, 1):
                    
            if not is_valid_json(file_path):
                print(f"Skipping empty or invalid file: {file_path}")
                continue  # Skip empty or malformed files

            with open(file_path, "r") as f:
                json_content = json.load(f)
            
            vehicle_positions = {}
            pedestrian_positions = {}
            
            extract_positions(json_content.get("past_vehicle_tracks", []), vehicle_positions)
            extract_positions(json_content.get("past_pedestrian_tracks", []), pedestrian_positions)

            # Count unique track IDs for vehicles and pedestrians
            num_vehicles = len(set(vehicle_positions.keys()))
            num_pedestrians = len(set(pedestrian_positions.keys()))
            
            all_frames = set()
            for time_key in json_content.get("past_vehicle_tracks", [])[0].keys():
                all_frames.add(time_key)
            for time_key in json_content.get("past_pedestrian_tracks", [])[0].keys():
                all_frames.add(time_key)
            
            flag = 0 # mark


            if num_vehicles >= 5 * num_pedestrians:
                continue

            for frame in all_frames:
                if flag:
                    break # search till a risky moment found
                for vehicle_id, vehicle_path in vehicle_positions.items():
                    if flag:
                        break # search till a risky moment found

                    vehicle_position, vehicle_velocity = get_position_and_velocity(vehicle_path, frame)
                    if not vehicle_position or not vehicle_velocity:
                        continue
        
                    for pedestrian_id, pedestrian_path in pedestrian_positions.items():
                        if flag:
                            break # search till a risky moment found

                        pedestrian_position, pedestrian_velocity = get_position_and_velocity(pedestrian_path, frame)
                        if not pedestrian_position or not pedestrian_velocity:
                            continue

                        # calculate TTC
                        t1, t2 = calculate_TTC(vehicle_position, pedestrian_position,vehicle_velocity, pedestrian_velocity)
            
                        # check TTC in range of potential of risk
                        if t1 is None or t2 is None:
                            continue

                        angle_delta = calculate_angle(vehicle_velocity, pedestrian_velocity)

                        if ( t1>0 and t2>0 and abs(t1-t2) <= TTC_MAX and
                            angle_delta >= 90-ANGLE_THRESHOLD and angle_delta <= 90+ANGLE_THRESHOLD ):

                            # Get all available time frames for this pedestrian
                            available_time_frames = [int(t.replace("time_", "")) for t, _, _, _, _ in pedestrian_path]
                            
                            future_time_start = min(max(available_time_frames),int(frame.split("_")[1]) + 1)  # Adjust to last valid time step
                            future_time_end = min(max(available_time_frames),int(frame.split("_")[1]) + 6)  # Adjust to last valid time step

                            # Check acceleration over the next 5 seconds
                            acceleration_flag = 0
                            
                            future_time_start = int(frame.split("_")[1]) + 1  # Start from the next time step
                            future_time_end = future_time_start + 5  # Check for the next 5 seconds

                            for future_time in range(future_time_start, future_time_end + 1):
                                future_pedestrian_position, future_pedestrian_velocity = get_position_and_velocity(
                                                                              pedestrian_path, f"time_{future_time}")

                                if future_pedestrian_velocity:
                                    # Compute speed change magnitude
                                    delta_v_x = future_pedestrian_velocity[0] - pedestrian_velocity[0]
                                    delta_v_y = future_pedestrian_velocity[1] - pedestrian_velocity[1]
                                    delta_v = (delta_v_x**2 + delta_v_y**2) ** 0.5  # Speed change magnitude
           
                                    if delta_v >= ACCELERATION_THRESHOLD:  # Define ACCELERATION_THRESHOLD
                                        acceleration_flag = 1
                                        break

                            # Check for a significant turn over the next 5 seconds
                            turn_flag = 0  # Initialize turn flag
                            for future_time in range(future_time_start, future_time_end + 1):
                                future_pedestrian_position, future_pedestrian_velocity = get_position_and_velocity(
                                                                                    pedestrian_path, f"time_{future_time}")

                                if future_pedestrian_velocity:
                                    turn_angle = calculate_angle(pedestrian_velocity, future_pedestrian_velocity)
                                    if turn_angle >= TURN_THRESHOLD:  # Define TURN_THRESHOLD (e.g., 30 degrees)
                                        turn_flag = 1
                                        break

                            # Check if speed exceeds threshold at any point in the trajectory
                            speed_exceed_flag = 0
                            for (_, _, _, vx, vy) in pedestrian_path:
                                speed = (vx**2 + vy**2) ** 0.5  # Compute speed magnitude
                                if speed > PEDES_SPEED_THRESHOLD:  # Pedestrian Speed Threshold
                                    speed_exceed_flag = 1
                                    break

                            if acceleration_flag or  turn_flag or speed_exceed_flag:
                                flag = 1
                                break


                        
            if flag:
                interaction_files.append(file_path)  
            print(f"    Processing file {file_idx}/{total_files_in_folder}: {file_path}")

        folder_name = os.path.basename(root)
        output_file_path = os.path.join(output_folder, f"{folder_name}_interaction_results.txt")
        
        with open(output_file_path, 'w') as out_f:
            out_f.write(f"Results for folder: {root}\n")
            if interaction_files:
                for file_name in interaction_files:
                    out_f.write(f"{file_name}\n")
            else:
                out_f.write("No interactions found in this folder.\n")
            interaction_files.clear()

root_folder_path = r"E:\BJTU-Thesis\data\train_json"
output_folder_path = r"E:\BJTU-Thesis\data\itr_json"

os.makedirs(output_folder_path, exist_ok=True)
process_all_files(root_folder_path, output_folder_path)
