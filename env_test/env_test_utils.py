import os
import numpy as np
import random

import matplotlib.pyplot as plt

def create_directories(formatted_time, render_modes, mode="block_traj"):
    if mode == "block_traj":
        output_dir = f'outputs standalone/{formatted_time}_{render_modes}'
    elif mode == "mujoco_test":
        output_dir = f'outputs mujoco test/{formatted_time}_{render_modes}'

    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def log_simulation_results(release_time, block_release_pos, block_release_orientation, 
                           block_release_transvel, block_release_angvel, touch_ground_time, 
                           block_touch_ground_position, block_touch_ground_orientation):
    print("="*91)
    print(f"Block release time: {release_time}")
    print(f"Block release position: {block_release_pos}")
    print(f"Block release orientation: {block_release_orientation}")
    print(f"Block translational release velocity: {block_release_transvel}")
    print(f"Block angular release velocity: {block_release_angvel}")
    print("-"*91)
    print(f"Block touch the ground time: {touch_ground_time}")
    print(f"Position when the block touched the ground: {block_touch_ground_position}")
    print(f"Orientation when the block touched the ground: {block_touch_ground_orientation}")

def plot_joints_data(output_dir, time_values, qpos_values, qvel_values, ctrl_values=None):
    qpos_values = np.array(qpos_values)
    qvel_values = np.array(qvel_values)
    num_joints = qpos_values.shape[1]

    # Plot joint positions
    fig1, axs1 = plt.subplots(num_joints, 1, figsize=(10, 12))
    for i in range(num_joints):
        axs1[i].plot(time_values, qpos_values[:, i], label=f"Joint {i+1} Position")
        axs1[i].set_ylabel(f"qpos {i+1}")
        axs1[i].legend()
        axs1[i].grid(True)
        # Get min and max values
        min_val = np.min(qpos_values[:, i])
        max_val = np.max(qpos_values[:, i])
        # Plot horizontal lines
        axs1[i].axhline(min_val, color='blue', linestyle='--', linewidth=0.8)
        axs1[i].axhline(max_val, color='red', linestyle='--', linewidth=0.8)
        # Annotate values
        axs1[i].text(time_values[0], min_val, f"{min_val:.2f}", color='blue', verticalalignment='bottom')
        axs1[i].text(time_values[0], max_val, f"{max_val:.2f}", color='red', verticalalignment='bottom')
    axs1[0].set_title("Joint Positions over Time")
    axs1[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_positions_vs_time.png'))

    # Plot joint velocities
    fig2, axs2 = plt.subplots(num_joints, 1, figsize=(10, 12))
    for i in range(num_joints):
        axs2[i].plot(time_values, qvel_values[:, i], label=f"Joint {i+1} Velocity", color='orange')
        axs2[i].set_ylabel(f"qvel {i+1}")
        axs2[i].legend()
        axs2[i].grid(True)
        # Get min and max values
        min_val = np.min(qvel_values[:, i])
        max_val = np.max(qvel_values[:, i])
        # Plot horizontal lines
        axs2[i].axhline(min_val, color='blue', linestyle='--', linewidth=0.8)
        axs2[i].axhline(max_val, color='red', linestyle='--', linewidth=0.8)
        # Annotate values
        axs2[i].text(time_values[0], min_val, f"{min_val:.2f}", color='blue', verticalalignment='bottom')
        axs2[i].text(time_values[0], max_val, f"{max_val:.2f}", color='red', verticalalignment='bottom')
    axs2[0].set_title("Joint Velocities over Time")
    axs2[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_velocities_vs_time.png'))

    # Plot control signals (if provided)
    if ctrl_values is not None:
        ctrl_values = np.array(ctrl_values)
        fig3, axs3 = plt.subplots(num_joints, 1, figsize=(10, 12))
        for i in range(num_joints):
            axs3[i].plot(time_values, ctrl_values[:, i], label=f"Joint {i+1} Control", color='green')
            axs3[i].set_ylabel(f"ctrl {i+1}")
            axs3[i].legend()
            axs3[i].grid(True)
            # Get min and max values
            min_val = np.min(ctrl_values[:, i])
            max_val = np.max(ctrl_values[:, i])
            # Plot horizontal lines
            axs3[i].axhline(min_val, color='blue', linestyle='--', linewidth=0.8)
            axs3[i].axhline(max_val, color='red', linestyle='--', linewidth=0.8)
            # Annotate values
            axs3[i].text(time_values[0], min_val, f"{min_val:.2f}", color='blue', verticalalignment='bottom')
            axs3[i].text(time_values[0], max_val, f"{max_val:.2f}", color='red', verticalalignment='bottom')
        axs3[0].set_title("Joint Controls over Time")
        axs3[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_controls_vs_time.png'))



def plot_block_data(output_dir, times, block_orientations_quat, desired_orientations_quat, block_orientations_euler, 
                    xfrc_applied_data, qfrc_applied_data, block_positions, block_trans_vels, block_ang_vels, 
                    flipped_time=None, touch_ground_time=None):

    block_orientations_quat = np.array(block_orientations_quat)
    desired_orientations_quat = np.array(desired_orientations_quat)
    block_orientations_euler = np.array(block_orientations_euler)
    block_positions = np.array(block_positions)
    block_trans_vels = np.array(block_trans_vels)
    block_ang_vels = np.array(block_ang_vels)
    
    # Plot 1: Block Angular Velocities, Block Position, Block Translational Velocities, Block Orientation Euler Angles
    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 16))
    
    # Block Angular Velocities
    axs1[0].plot(times, block_ang_vels[:, 0], label="Ang Vel x", color='lime')
    axs1[0].plot(times, block_ang_vels[:, 1], label="Ang Vel y", color='pink')
    axs1[0].plot(times, block_ang_vels[:, 2], label="Ang Vel z", color='navy')
    axs1[0].set_ylabel("Angular Velocities")
    axs1[0].legend()
    axs1[0].grid(True)
    if flipped_time is not None:
        axs1[0].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs1[0].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")

    # Block Positions
    axs1[1].plot(times, block_positions[:, 0], label="Pos x", color='blue')
    axs1[1].plot(times, block_positions[:, 1], label="Pos y", color='green')
    axs1[1].plot(times, block_positions[:, 2], label="Pos z", color='red')
    axs1[1].set_ylabel("Positions")
    axs1[1].legend()
    axs1[1].grid(True)
    if flipped_time is not None:
        axs1[1].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs1[1].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")

    # Block Translational Velocities
    axs1[2].plot(times, block_trans_vels[:, 0], label="Trans Vel x", color='cyan')
    axs1[2].plot(times, block_trans_vels[:, 1], label="Trans Vel y", color='magenta')
    axs1[2].plot(times, block_trans_vels[:, 2], label="Trans Vel z", color='yellow')
    axs1[2].set_ylabel("Translational Velocities")
    axs1[2].legend()
    axs1[2].grid(True)
    if flipped_time is not None:
        axs1[2].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs1[2].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")

    # Block Orientation (Euler Angles)
    axs1[3].plot(times, block_orientations_euler[:, 0], label="Euler z (Yaw)", color='purple')
    axs1[3].plot(times, block_orientations_euler[:, 1], label="Euler y (Pitch)", color='teal')
    axs1[3].plot(times, block_orientations_euler[:, 2], label="Euler x (Roll)", color='brown')
    axs1[3].set_xlabel("Time (s)")
    axs1[3].set_ylabel("Euler Angles (degrees)")
    axs1[3].legend()
    axs1[3].grid(True)
    if flipped_time is not None:
        axs1[3].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs1[3].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    fig1.suptitle("Block Angular Velocities, Positions, Translational Velocities, and Euler Angles Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'block_state_vs_time.png'))

    # Plot 2: Block Orientation Quaternion
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 12))
    axs2[0].plot(times, block_orientations_quat[:, 0], label="Block Quaternion x", color='b')
    axs2[0].plot(times, desired_orientations_quat[:, 0], '--', label="Desired Quaternion x", color='b')
    axs2[0].set_ylabel("Quaternion x")
    axs2[0].legend()
    axs2[0].grid(True)
    if flipped_time is not None:
        axs2[0].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs2[0].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    axs2[1].plot(times, block_orientations_quat[:, 1], label="Block Quaternion y", color='orange')
    axs2[1].plot(times, desired_orientations_quat[:, 1], '--', label="Desired Quaternion y", color='orange')
    axs2[1].set_ylabel("Quaternion y")
    axs2[1].legend()
    axs2[1].grid(True)
    if flipped_time is not None:
        axs2[1].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs2[1].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    axs2[2].plot(times, block_orientations_quat[:, 2], label="Block Quaternion z", color='g')
    axs2[2].plot(times, desired_orientations_quat[:, 2], '--', label="Desired Quaternion z", color='g')
    axs2[2].set_ylabel("Quaternion z")
    axs2[2].legend()
    axs2[2].grid(True)
    if flipped_time is not None:
        axs2[2].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs2[2].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    axs2[3].plot(times, block_orientations_quat[:, 3], label="Block Quaternion w", color='r')
    axs2[3].plot(times, desired_orientations_quat[:, 3], '--', label="Desired Quaternion w", color='r')
    axs2[3].set_xlabel("Time (s)")
    axs2[3].set_ylabel("Quaternion w")
    axs2[3].legend()
    axs2[3].grid(True)
    if flipped_time is not None:
        axs2[3].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs2[3].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")    
    
    fig2.suptitle("Block Orientation (Quaternion Components) Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'block_orientation_quat_vs_time.png'))

    # Plot 3: Applied Forces
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8))
    axs3[0].plot(times, np.array(xfrc_applied_data))
    axs3[0].set_ylabel("xfrc_applied")
    axs3[0].legend(["x", "y", "z", "rx", "ry", "rz"])
    axs3[0].grid(True)
    if flipped_time is not None:
        axs3[0].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs3[0].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    axs3[1].plot(times, np.array(qfrc_applied_data))
    axs3[1].set_xlabel("Time (s)")
    axs3[1].set_ylabel("qfrc_applied[14:20]")
    axs3[1].legend([f"q_{i}" for i in range(14, 20)])
    axs3[1].grid(True)
    if flipped_time is not None:
        axs3[1].axvline(x=flipped_time, color='r', linestyle='--', label="Flipped Time")
    if touch_ground_time is not None:
        axs3[1].axvline(x=touch_ground_time, color='g', linestyle='--', label="Touch Ground Time")
    
    fig3.suptitle("Applied Forces Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'applied_forces_vs_time.png'))



def generate_random_block_params():
    block_mass = round(random.uniform(0.01, 0.3), 4)

    # Generate random sizes for block_size within their specified ranges
    length = round(random.uniform(0.02, 0.08), 4)  # largest dimension
    height = round(random.uniform(0.01, 0.04), 4)  # second largest dimension
    width = round(random.uniform(0.005, 0.02), 4)  # smallest dimension

    block_size = sorted([height, width, length], reverse=True)

    return block_mass, block_size

