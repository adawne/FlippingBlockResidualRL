import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first file
file1 = '../interpolated_trajectory.csv'  # Replace with the actual path of the first file
df1 = pd.read_csv(file1)

# Extract QPos and QVel from the first file
qpos1 = df1['QPos'].apply(lambda x: [float(val) for val in x.split(',')[:6]])  # Take first 6 QPos values
qpos1 = np.array(qpos1.tolist())  # Convert to a 2D numpy array
qvel1 = df1['QVel'].apply(lambda x: [float(val) for val in x.split(',')[:6]])  # Take first 6 QVel values
qvel1 = np.array(qvel1.tolist())  # Convert to a 2D numpy array

# Load the second file
file2 = '../../mpc_log.csv'  # Replace with the actual path of the second file
df2 = pd.read_csv(file2)

# Extract QPos and QVel from the second file
qpos2 = df2['QPos'].apply(lambda x: [float(val) for val in x.split(',')[:6]])  # Take first 6 QPos values
qpos2 = np.array(qpos2.tolist())  # Convert to a 2D numpy array
qvel2 = df2['QVel'].apply(lambda x: [float(val) for val in x.split(',')[:6]])  # Take first 6 QVel values
qvel2 = np.array(qvel2.tolist())  # Convert to a 2D numpy array

# Make sure all arrays have the same length
min_len = min(len(qpos1), len(qpos2))
qpos1, qpos2 = qpos1[:min_len], qpos2[:min_len]
qvel1, qvel2 = qvel1[:min_len], qvel2[:min_len]

# Calculate the deltas
delta_qpos = qpos1 - qpos2
delta_qvel = qvel1 - qvel2

# Get file names without path and extension
file1_name = os.path.splitext(os.path.basename(file1))[0]
file2_name = os.path.splitext(os.path.basename(file2))[0]

# Plotting QPos
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
time_indices = np.arange(len(qpos1))

for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time_indices, qpos1[:, i], label=f'{file1_name} Joint {i+1}', linestyle='--')
    axs[row, col].plot(time_indices, qpos2[:, i], label=f'{file2_name} Joint {i+1}', linestyle='-')
    axs[row, col].set_title(f'QPos for Joint {i+1}')
    axs[row, col].set_xlabel('Time Index')
    axs[row, col].set_ylabel('QPos')
    axs[row, col].legend()

plt.tight_layout()
plt.savefig('qpos_comparison.png')
plt.close()

# Plotting QVel
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time_indices, qvel1[:, i], label=f'{file1_name} Joint {i+1}', linestyle='--')
    axs[row, col].plot(time_indices, qvel2[:, i], label=f'{file2_name} Joint {i+1}', linestyle='-')
    axs[row, col].set_title(f'QVel for Joint {i+1}')
    axs[row, col].set_xlabel('Time Index')
    axs[row, col].set_ylabel('QVel')
    axs[row, col].legend()

plt.tight_layout()
plt.savefig('qvel_comparison.png')
plt.close()

# Plotting Delta QPos
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time_indices, delta_qpos[:, i], label=f'Delta QPos Joint {i+1}', linestyle=':')
    axs[row, col].set_title(f'Delta QPos for Joint {i+1}')
    axs[row, col].set_xlabel('Time Index')
    axs[row, col].set_ylabel('Delta QPos')
    axs[row, col].legend()

plt.tight_layout()
plt.savefig('delta_qpos_comparison.png')
plt.close()

# Plotting Delta QVel
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time_indices, delta_qvel[:, i], label=f'Delta QVel Joint {i+1}', linestyle=':')
    axs[row, col].set_title(f'Delta QVel for Joint {i+1}')
    axs[row, col].set_xlabel('Time Index')
    axs[row, col].set_ylabel('Delta QVel')
    axs[row, col].legend()

plt.tight_layout()
plt.savefig('delta_qvel_comparison.png')
plt.close()

print("Plots saved as 'qpos_comparison.png', 'qvel_comparison.png', 'delta_qpos_comparison.png', and 'delta_qvel_comparison.png'")
