import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first file (assuming QPos values are in a single column as comma-separated strings)
file1 = '../interpolated_trajectory.csv'  # Replace with the actual path of the first file
df1 = pd.read_csv(file1)

# Extract the first 6 QPos values from each row in the first file
qpos1 = df1['QPos'].apply(lambda x: [float(val) for val in x.split(',')[:6]])  # Only take the first 6 values
qpos1 = np.array(qpos1.tolist())  # Convert to a 2D numpy array

# Load the second file (assuming QPos values are in separate columns for each joint)
file2 = '../../mpc_state_log_new.csv'  # Replace with the actual path of the second file
df2 = pd.read_csv(file2)

# Extract QPos values from each joint column in the second file
qpos2 = [df2[f'Joint_{i+1}'].values for i in range(6)]  # Assuming columns are named 'Joint_1', 'Joint_2', ..., 'Joint_6'

# Convert qpos2 to a list of arrays for compatibility and stack them into a 2D array
qpos2 = np.stack(qpos2, axis=1)  # Convert list of arrays to a 2D array with shape (time, joints)

# Make sure both QPos arrays have the same length
if len(qpos1) != len(qpos2):
    min_len = min(len(qpos1), len(qpos2))
    qpos1 = qpos1[:min_len]
    qpos2 = qpos2[:min_len]

# Calculate the delta between the QPos values for each joint
delta_qpos = qpos1 - qpos2

# Plotting the delta for all joints
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid for subplots

for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(delta_qpos[:, i], label=f'Delta for Joint {i+1}')
    axs[row, col].set_title(f'Delta for Joint {i+1}')
    axs[row, col].set_xlabel('Time Index')
    axs[row, col].set_ylabel('Delta QPos')
    axs[row, col].legend()

plt.tight_layout()
plt.savefig('delta_qpos_all_joints_ctrl.png')
plt.show()
