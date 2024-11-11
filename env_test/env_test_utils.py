import numpy as np
import matplotlib.pyplot as plt



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



def plot_block_data(times, block_orientations_quat, desired_orientations_quat, block_orientations_euler, 
                    xfrc_applied_data, qfrc_applied_data, block_positions, block_trans_vels, block_ang_vels, 
                    flipped_time=None):

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
    
    # Block Positions
    axs1[1].plot(times, block_positions[:, 0], label="Pos x", color='blue')
    axs1[1].plot(times, block_positions[:, 1], label="Pos y", color='green')
    axs1[1].plot(times, block_positions[:, 2], label="Pos z", color='red')
    axs1[1].set_ylabel("Positions")
    axs1[1].legend()
    axs1[1].grid(True)
    if flipped_time is not None:
        axs1[1].axvline(x=flipped_time, color='r', linestyle='--')
    
    # Block Translational Velocities
    axs1[2].plot(times, block_trans_vels[:, 0], label="Trans Vel x", color='cyan')
    axs1[2].plot(times, block_trans_vels[:, 1], label="Trans Vel y", color='magenta')
    axs1[2].plot(times, block_trans_vels[:, 2], label="Trans Vel z", color='yellow')
    axs1[2].set_ylabel("Translational Velocities")
    axs1[2].legend()
    axs1[2].grid(True)
    if flipped_time is not None:
        axs1[2].axvline(x=flipped_time, color='r', linestyle='--')
    
    # Block Orientation (Euler Angles)
    axs1[3].plot(times, block_orientations_euler[:, 0], label="Euler z (Yaw)", color='purple')
    axs1[3].plot(times, block_orientations_euler[:, 1], label="Euler y (Pitch)", color='teal')
    axs1[3].plot(times, block_orientations_euler[:, 2], label="Euler x (Roll)", color='brown')
    axs1[3].set_xlabel("Time (s)")
    axs1[3].set_ylabel("Euler Angles (degrees)")
    axs1[3].legend()
    axs1[3].grid(True)
    if flipped_time is not None:
        axs1[3].axvline(x=flipped_time, color='r', linestyle='--')
    
    fig1.suptitle("Block Angular Velocities, Positions, Translational Velocities, and Euler Angles Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('block_state_vs_time.png')

    # Plot 2: Block Orientation Quaternion
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 12))
    axs2[0].plot(times, block_orientations_quat[:, 0], label="Block Quaternion x", color='b')
    axs2[0].plot(times, desired_orientations_quat[:, 0], '--', label="Desired Quaternion x", color='b')
    axs2[0].set_ylabel("Quaternion x")
    axs2[0].legend()
    axs2[0].grid(True)
    if flipped_time is not None:
        axs2[0].axvline(x=flipped_time, color='r', linestyle='--')
    
    axs2[1].plot(times, block_orientations_quat[:, 1], label="Block Quaternion y", color='orange')
    axs2[1].plot(times, desired_orientations_quat[:, 1], '--', label="Desired Quaternion y", color='orange')
    axs2[1].set_ylabel("Quaternion y")
    axs2[1].legend()
    axs2[1].grid(True)
    if flipped_time is not None:
        axs2[1].axvline(x=flipped_time, color='r', linestyle='--')
    
    axs2[2].plot(times, block_orientations_quat[:, 2], label="Block Quaternion z", color='g')
    axs2[2].plot(times, desired_orientations_quat[:, 2], '--', label="Desired Quaternion z", color='g')
    axs2[2].set_ylabel("Quaternion z")
    axs2[2].legend()
    axs2[2].grid(True)
    if flipped_time is not None:
        axs2[2].axvline(x=flipped_time, color='r', linestyle='--')
    
    axs2[3].plot(times, block_orientations_quat[:, 3], label="Block Quaternion w", color='r')
    axs2[3].plot(times, desired_orientations_quat[:, 3], '--', label="Desired Quaternion w", color='r')
    axs2[3].set_xlabel("Time (s)")
    axs2[3].set_ylabel("Quaternion w")
    axs2[3].legend()
    axs2[3].grid(True)
    if flipped_time is not None:
        axs2[3].axvline(x=flipped_time, color='r', linestyle='--')
    
    fig2.suptitle("Block Orientation (Quaternion Components) Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('block_orientation_quat_vs_time.png')

    # Plot 3: Applied Forces
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8))
    axs3[0].plot(times, np.array(xfrc_applied_data))
    axs3[0].set_ylabel("xfrc_applied")
    axs3[0].legend(["x", "y", "z", "rx", "ry", "rz"])
    axs3[0].grid(True)
    if flipped_time is not None:
        axs3[0].axvline(x=flipped_time, color='r', linestyle='--')
    
    axs3[1].plot(times, np.array(qfrc_applied_data))
    axs3[1].set_xlabel("Time (s)")
    axs3[1].set_ylabel("qfrc_applied[14:20]")
    axs3[1].legend([f"q_{i}" for i in range(14, 20)])
    axs3[1].grid(True)
    if flipped_time is not None:
        axs3[1].axvline(x=flipped_time, color='r', linestyle='--')
    
    fig3.suptitle("Applied Forces Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('applied_forces_vs_time.png')
