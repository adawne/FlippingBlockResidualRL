import json
import numpy as np

def analyze_simulation_results(file_path, output_file):
    with open(file_path, 'r') as f:
        data = json.load(f)

    min_norm_diff = float('inf')
    min_norm_entry = None

    min_velocity_diff_x = float('inf')
    min_velocity_diff_y = float('inf')
    min_velocity_diff_z = float('inf')

    config_min_x = None
    config_min_y = None
    config_min_z = None

    deltas_min_x = None
    deltas_min_y = None
    deltas_min_z = None

    deltas_min_norm = None

    for entry in data:
        block_release_transvel = np.array(entry["block_release_transvel"])
        fsm_release_ee_velocity = np.array(entry["fsm_release_ee_velocity"])

        norm_diff = np.linalg.norm(block_release_transvel - fsm_release_ee_velocity)

        if norm_diff < min_norm_diff:
            min_norm_diff = norm_diff
            min_norm_entry = entry["current_config"]
            deltas_min_norm = block_release_transvel - fsm_release_ee_velocity

        velocity_diff_x = abs(block_release_transvel[0] - fsm_release_ee_velocity[0])
        velocity_diff_y = abs(block_release_transvel[1] - fsm_release_ee_velocity[1])
        velocity_diff_z = abs(block_release_transvel[2] - fsm_release_ee_velocity[2])

        if velocity_diff_x < min_velocity_diff_x:
            min_velocity_diff_x = velocity_diff_x
            config_min_x = entry["current_config"]
            deltas_min_x = block_release_transvel[0] - fsm_release_ee_velocity[0]
        
        if velocity_diff_y < min_velocity_diff_y:
            min_velocity_diff_y = velocity_diff_y
            config_min_y = entry["current_config"]
            deltas_min_y = block_release_transvel[1] - fsm_release_ee_velocity[1]
        
        if velocity_diff_z < min_velocity_diff_z:
            min_velocity_diff_z = velocity_diff_z
            config_min_z = entry["current_config"]
            deltas_min_z = block_release_transvel[2] - fsm_release_ee_velocity[2]

    results = {
        "smallest_norm_difference": {
            "norm_value": min_norm_diff,
            "configuration": min_norm_entry,
            "deltas": deltas_min_norm.tolist()
        },
        "smallest_velocity_difference_x": {
            "velocity_difference": min_velocity_diff_x,
            "configuration": config_min_x,
            "delta_x": deltas_min_x
        },
        "smallest_velocity_difference_y": {
            "velocity_difference": min_velocity_diff_y,
            "configuration": config_min_y,
            "delta_y": deltas_min_y
        },
        "smallest_velocity_difference_z": {
            "velocity_difference": min_velocity_diff_z,
            "configuration": config_min_z,
            "delta_z": deltas_min_z
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Smallest norm difference: {min_norm_diff}, Deltas: {deltas_min_norm}")
    print(f"Smallest velocity difference on X-axis: {min_velocity_diff_x}, Delta X: {deltas_min_x}")
    print(f"Smallest velocity difference on Y-axis: {min_velocity_diff_y}, Delta Y: {deltas_min_y}")
    print(f"Smallest velocity difference on Z-axis: {min_velocity_diff_z}, Delta Z: {deltas_min_z}")

analyze_simulation_results('all_simulation_results_225_010.json', 'params_mass010_clamp225.json')


