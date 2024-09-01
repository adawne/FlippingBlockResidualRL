import os
import cv2
import mujoco
import matplotlib.pyplot as plt
import numpy as np


class SimulationRenderer:
    def __init__(self, model, data, output_dir, render_mode=None, contact_vis=None, framerate=60):
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.render_mode = render_mode
        self.contact_vis = contact_vis
        self.framerate = framerate
        self.frames = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)

        if self.render_mode is not None:
            if self.render_mode == 'livecam':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.renderer = mujoco.Renderer(self.model, height=1024, width=1440)
                self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(os.path.join(self.output_dir, 'random_test.mp4'), self.fourcc, self.framerate, (1440, 1024))

            if self.contact_vis is not None:
                self.options = mujoco.MjvOption()
                mujoco.mjv_defaultOption(self.options)
                self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

                # tweak scales of contact visualization elements
                self.model.vis.scale.contactwidth = 0.1
                self.model.vis.scale.contactheight = 0.03
                self.model.vis.scale.forcewidth = 0.05
                self.model.vis.map.force = 0.3

    def render_frame(self, time):
        if self.render_mode == 'livecam':
            self.viewer.sync()
        else:
            if len(self.frames) < time * self.framerate:
                if self.render_mode == 'video_top':
                    self.camera.distance = 3
                    self.camera.azimuth = 190
                    self.camera.elevation = -45
                elif self.render_mode == 'video_side':
                    self.camera.distance = 3
                    self.camera.azimuth = 130
                    self.camera.elevation = -25
                elif self.render_mode == 'video_front':
                    self.camera.distance = 3
                    self.camera.azimuth = 180
                    self.camera.elevation = -50

                if self.contact_vis is not None:
                    self.renderer.update_scene(self.data, self.camera, self.options)
                else:
                    self.renderer.update_scene(self.data, self.camera)
                pixels = self.renderer.render()
                self.frames.append(pixels)
                self.out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    def take_screenshot(self, time_step, file_name='screenshot'):
        if self.render_mode != 'livecam':
            file_path = os.path.join(self.output_dir, f"{file_name}_timestep_{time_step:.4f}.png")
            pixels = self.renderer.render()
            cv2.imwrite(file_path, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            print(f"Screenshot taken at timestep {time_step:.4f} and saved to {file_path}")
        else:
            pass

    def close(self):
        if self.render_mode == 'livecam':
            self.viewer.close()
        else:
            if hasattr(self, 'out') and self.out.isOpened():
                self.out.release()
            
            self.frames.clear()
            if hasattr(self, 'renderer'):
                del self.renderer



def plot_block_pose(output_dir, release_time, landing_time_pred, touch_ground_time, steady_time, time_hist, block_position_hist, block_orientation_hist, block_trans_vel_hist, block_ang_vel_hist):
    # Process the data for plotting
    block_position_hist = list(zip(*block_position_hist))  # Transpose to separate x, y, z components
    block_orientation_hist = list(zip(*block_orientation_hist))  # Transpose to separate x, y, z components
    block_trans_vel_hist = list(zip(*block_trans_vel_hist))  # Transpose to separate x, y, z components
    block_ang_vel_hist = list(zip(*block_ang_vel_hist))  # Transpose to separate x, y, z components

    plt.figure(figsize=(16, 12))

    # Plotting block position history
    plt.subplot(4, 1, 1)
    plt.plot(time_hist, block_position_hist[0], label='X position')
    plt.plot(time_hist, block_position_hist[1], label='Y position')
    plt.plot(time_hist, block_position_hist[2], label='Z position')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Block Position History')
    plt.legend()

    # Plotting block orientation history
    plt.subplot(4, 1, 2)
    plt.plot(time_hist, block_orientation_hist[0], label='X orientation')
    plt.plot(time_hist, block_orientation_hist[1], label='Y orientation')
    plt.plot(time_hist, block_orientation_hist[2], label='Z orientation')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation')
    plt.title('Block Orientation History')
    plt.legend()

    # Plotting translational velocity qvel[14:17]
    plt.subplot(4, 1, 3)
    plt.plot(time_hist, block_trans_vel_hist[0], label='Vx')
    plt.plot(time_hist, block_trans_vel_hist[1], label='Vy')
    plt.plot(time_hist, block_trans_vel_hist[2], label='Vz')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Translational Velocity qvel[14:17]')
    plt.legend()

    # Plotting angular velocity qvel[17:20]
    plt.subplot(4, 1, 4)
    plt.plot(time_hist, block_ang_vel_hist[0], label='Omega x')
    plt.plot(time_hist, block_ang_vel_hist[1], label='Omega y')
    plt.plot(time_hist, block_ang_vel_hist[2], label='Omega z')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity qvel[17:20]')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'block_flying_poses.png'))


def plot_joint_velocities(output_dir, release_time, time_hist, joint_vel_hist, target_joint_vel_hist):
    joint_vel_hist_np = np.array(joint_vel_hist)
    target_joint_vel_hist_np = np.array(target_joint_vel_hist)
    time_hist_np = np.array(time_hist)

    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

    fig, axs = plt.subplots(6, 1, figsize=(10, 15))  # 6 rows, 1 column of subplots

    for i in range(joint_vel_hist_np.shape[1]):
        axs[i].plot(time_hist_np, joint_vel_hist_np[:, i], label=f'Real {joint_names[i]} Velocity')
        axs[i].plot(time_hist_np, target_joint_vel_hist_np[:, i], linestyle='--', label=f'Target {joint_names[i]} Velocity')
        
        # Plot the vertical line at release time
        axs[i].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Velocity (rad/s)')
        axs[i].set_title(f'{joint_names[i]} Velocity Over Time')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_velocities_subplot.png'))

def plot_velocity_comparison(output_dir, release_time, time_hist, end_effector_vel_hist, block_vel_hist):
    # Ignore the first element of each list
    time_hist_np = np.array(time_hist[1:])
    end_effector_vel_hist_np = np.array(end_effector_vel_hist[1:])  # Shape: (num_samples-1, 3)
    block_vel_hist_np = np.array(block_vel_hist[1:])  # Shape: (num_samples-1, 3)

    velocity_components = ['X', 'Y', 'Z']

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))  # 3 rows, 2 columns of subplots (position and discrepancy for each axis)

    for i in range(3):
        axs[i, 0].plot(time_hist_np, end_effector_vel_hist_np[:, i], label=f'End Effector {velocity_components[i]} Velocity')
        axs[i, 0].plot(time_hist_np, block_vel_hist_np[:, i], linestyle='--', label=f'Block {velocity_components[i]} Velocity')
        
        axs[i, 0].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Velocity (m/s)')
        axs[i, 0].set_title(f'{velocity_components[i]} Velocity Over Time')
        axs[i, 0].legend()

        velocity_discrepancy = end_effector_vel_hist_np[:, i] - block_vel_hist_np[:, i]
        axs[i, 1].plot(time_hist_np, velocity_discrepancy, color='purple', label=f'{velocity_components[i]} Velocity Discrepancy')
        
        axs[i, 1].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Discrepancy (m/s)')
        axs[i, 1].set_title(f'{velocity_components[i]} Velocity Discrepancy (End Effector - Block)')
        axs[i, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'end_effector_block_velocity_comparison.png'))

def plot_discrepancy_vs_mass(masses, time_discrepancies, angle_discrepancies, block_release_ver_velocity, filename='discrepancy_vs_mass.png'):
    plt.figure(figsize=(8, 18))  # Adjusting the figure size for vertical layout

    # Plot Time Discrepancy
    plt.subplot(3, 1, 1)
    plt.plot(masses, time_discrepancies, 'o-', color='blue', label='Time Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Time Discrepancy (s)')
    plt.title('Mass vs Time Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Angle Discrepancy for X, Y, Z axes
    plt.subplot(3, 1, 2)
    angle_discrepancies = np.array(angle_discrepancies)  # Convert to NumPy array for easy slicing
    plt.plot(masses, angle_discrepancies[:, 0], 'o-', color='red', label='Angle Discrepancy X')
    plt.plot(masses, angle_discrepancies[:, 1], 'o-', color='green', label='Angle Discrepancy Y')
    plt.plot(masses, angle_discrepancies[:, 2], 'o-', color='orange', label='Angle Discrepancy Z')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Angle Discrepancy (radians)')
    plt.title('Mass vs Angle Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Block Release Vertical Velocity
    plt.subplot(3, 1, 3)
    plt.plot(masses, block_release_ver_velocity, 'o-', color='purple', label='Block Release Vertical Velocity')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.title('Mass vs Block Release Vertical Velocity')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()