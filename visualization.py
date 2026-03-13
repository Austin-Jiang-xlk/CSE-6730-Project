import os
import json
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.cm as cm

# Set paths
itr_json_path = r"E:\BJTU-Thesis\data\itr_json"
output_path = r"E:\BJTU-Thesis\data\trajectory_figure"
os.makedirs(output_path, exist_ok=True)

# Function to check if a line contains a valid JSON file path
def is_valid_json_line(line):
    return line.strip().endswith(".json")

# Iterate through all _interaction_results.txt files in itr_json_path
for filename in os.listdir(itr_json_path):
    if not filename.endswith("_interaction_results.txt"):
        continue  # Only process interaction results files

    file_path = os.path.join(itr_json_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract JSON file paths (skip first line, stop at non-JSON lines)
    json_files = []
    for line in lines[1:]:  # Start from second line
        if not is_valid_json_line(line):
            break  # Stop processing when encountering a non-JSON line
        json_files.append(line.strip())

    # Process each valid JSON file
    for json_file_path in json_files:
        json_file_name = os.path.basename(json_file_path)
        json_id = os.path.splitext(json_file_name)[0]

        # Read JSON file
        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
        except Exception as e:
            print(f"Failed to read JSON file {json_file_path}: {e}")
            continue  # Skip this file if unreadable

        vehicle_trajectories = {}
        pedestrian_trajectories = {}

        past_vehicle_tracks = data.get("past_vehicle_tracks", [])
        future_vehicle_tracks = data.get("future_vehicle_tracks", [])
        past_pedestrian_tracks = data.get("past_pedestrian_tracks", [])
        future_pedestrian_tracks = data.get("future_pedestrian_tracks", [])

        # Extract vehicle trajectories
        for timestamp_data in past_vehicle_tracks + future_vehicle_tracks:
            for timestamp, vehicles in timestamp_data.items():
                for vehicle in vehicles:
                    track_id = vehicle["track_id"]
                    if track_id not in vehicle_trajectories:
                        vehicle_trajectories[track_id] = {"x": [], "y": [], "t": []}
                    vehicle_trajectories[track_id]["x"].append(vehicle["position"]["x"])
                    vehicle_trajectories[track_id]["y"].append(vehicle["position"]["y"])
                    vehicle_trajectories[track_id]["t"].append(int(timestamp.split("_")[1]))

        # Extract pedestrian trajectories
        for timestamp_data in past_pedestrian_tracks + future_pedestrian_tracks:
            for timestamp, pedestrians in timestamp_data.items():
                for pedestrian in pedestrians:
                    track_id = pedestrian["track_id"]
                    if track_id not in pedestrian_trajectories:
                        pedestrian_trajectories[track_id] = {"x": [], "y": [], "t": []}
                    pedestrian_trajectories[track_id]["x"].append(pedestrian["position"]["x"])
                    pedestrian_trajectories[track_id]["y"].append(pedestrian["position"]["y"])
                    pedestrian_trajectories[track_id]["t"].append(int(timestamp.split("_")[1]))

        # Count the number of vehicles and pedestrians
        num_vehicles = len(vehicle_trajectories)
        num_pedestrians = len(pedestrian_trajectories)

        # Create trajectory visualization
        plt.figure(figsize=(8, 6))

        # Plot vehicle trajectories (blue with time gradient)
        for track_id, traj in vehicle_trajectories.items():
            time_norm = np.array(traj["t"]) / max(traj["t"]) if traj["t"] else np.array([0])
            colors = cm.Blues(time_norm)
            plt.scatter(traj["x"], traj["y"], c=colors, s=10)

        # Plot pedestrian trajectories (red with time gradient)
        for track_id, traj in pedestrian_trajectories.items():
            time_norm = np.array(traj["t"]) / max(traj["t"]) if traj["t"] else np.array([0])
            colors = cm.Reds(time_norm)
            plt.scatter(traj["x"], traj["y"], c=colors, s=10)

        # Set title and labels
        plt.title(f"Trajectory Visualization - {json_id}\nVehicles: {num_vehicles}, Pedestrians: {num_pedestrians}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.colorbar(label="Time Progression")
        plt.grid(True)

        # Save the figure
        output_filename = os.path.join(output_path, f"{json_id}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close()

        print(f"Created and saved trajectory figure: {output_filename}")
