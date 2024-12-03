import os
import numpy as np
import pandas as pd

def calculate_average_delta(file1, file2):
    """
    Calculate the average delta QPos, delta QVel, and combined delta between two files.
    
    Args:
        file1 (str): Path to the first file (e.g., MPC trajectory file).
        file2 (str): Path to the second file (comparison approach file).
    
    Returns:
        avg_delta_qpos (float): Average delta QPos across all joints and timesteps.
        avg_delta_qvel (float): Average delta QVel across all joints and timesteps.
        avg_combined_delta (float): Average of QPos and QVel deltas combined.
    """
    # Load the first file
    df1 = pd.read_csv(file1)
    qpos1 = df1['QPos'].apply(lambda x: [float(val) for val in x.split(',')[:6]])
    qpos1 = np.array(qpos1.tolist())  # Convert to a 2D numpy array
    qvel1 = df1['QVel'].apply(lambda x: [float(val) for val in x.split(',')[:6]])
    qvel1 = np.array(qvel1.tolist())  # Convert to a 2D numpy array

    # Load the second file
    df2 = pd.read_csv(file2)
    qpos2 = df2['QPos'].apply(lambda x: [float(val) for val in x.split(',')[:6]])
    qpos2 = np.array(qpos2.tolist())  # Convert to a 2D numpy array
    qvel2 = df2['QVel'].apply(lambda x: [float(val) for val in x.split(',')[:6]])
    qvel2 = np.array(qvel2.tolist())  # Convert to a 2D numpy array

    # Make sure all arrays have the same length
    min_len = min(len(qpos1), len(qpos2))
    qpos1, qpos2 = qpos1[:min_len], qpos2[:min_len]
    qvel1, qvel2 = qvel1[:min_len], qvel2[:min_len]

    # Calculate deltas
    delta_qpos = np.abs(qpos1 - qpos2)  # Absolute difference
    delta_qvel = np.abs(qvel1 - qvel2)  # Absolute difference

    # Calculate average delta for QPos, QVel, and combined
    avg_delta_qpos = np.mean(delta_qpos)
    avg_delta_qvel = np.mean(delta_qvel)
    avg_combined_delta = (avg_delta_qpos + avg_delta_qvel) / 2  # Combined average

    return avg_delta_qpos, avg_delta_qvel, avg_combined_delta



# Define the files
mpc_file = '../interpolated_trajectory.csv'  # Replace with actual MPC file
# approach_files = {
#     'Ctrl': 'Ctrl/mpc_log.csv',  # Replace with actual file
#     'QPos': 'QPos/mpc_log.csv',  # Replace with actual file
#     'QVel': 'QVel/mpc_log.csv',  # Replace with actual file
# }

approach_files = {
    'Ctrl': 'Ctrl/mpc_log.csv',  # Replace with actual file
    'QPos': 'QPos/mpc_log.csv',  # Replace with actual file
    'QVel': 'QVel/mpc_log.csv',  # Replace with actual file
}

# Compare QPos and QVel across approaches
results = {}
for approach, file in approach_files.items():
    avg_delta_qpos, avg_delta_qvel, avg_combined_delta = calculate_average_delta(mpc_file, file)
    results[approach] = {
        'Avg Delta QPos': avg_delta_qpos,
        'Avg Delta QVel': avg_delta_qvel,
        'Avg Combined Delta': avg_combined_delta
    }

# Display the results
results_df = pd.DataFrame(results).T
print("Comparison of Average Deltas Across Approaches:")
print(results_df)

