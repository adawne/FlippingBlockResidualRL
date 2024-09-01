import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2

import matplotlib.pyplot as plt
from datetime import datetime

from mujoco_utils import *
from visual_utils import *
from scene_builder import *
from finite_state_machine import *


def check_physical_assumptions(release_time, touch_ground_time, block_release_ver_velocity, block_release_orientation, touch_ground_orientation, time_hist, block_ang_vel_hist, g=9.81):
    time_in_air_exp = touch_ground_time - release_time
    time_in_air_theory = 2 * block_release_ver_velocity / g
    time_discrepancy = np.abs(time_in_air_exp - time_in_air_theory)
    time_discrepancy_percentage = (time_discrepancy / time_in_air_theory) * 100
    
    in_air_indices = [i for i, t in enumerate(time_hist) if release_time <= t <= touch_ground_time]

    if in_air_indices:
        omega_exp = np.mean(np.array(block_ang_vel_hist)[in_air_indices], axis=0)
    else:
        omega_exp = np.zeros_like(block_release_orientation)
    
    theta_final_theory = omega_exp + (block_release_orientation * time_in_air_exp)
    
    # Unwrap the angles to handle discontinuities at ±pi
    theta_final_theory = np.unwrap(theta_final_theory)
    touch_ground_orientation = np.unwrap(touch_ground_orientation)
    
    # Normalize angles to the range [0, pi]
    theta_final_theory = np.mod(theta_final_theory, np.pi)
    touch_ground_orientation = np.mod(touch_ground_orientation, np.pi)
    
    theta_final_discrepancy = theta_final_theory - touch_ground_orientation
    theta_final_discrepancy_percentage = (theta_final_discrepancy / theta_final_theory) * 100 if np.any(theta_final_theory) else 0

    print("="*91)
    print("Testing Physical Assumptions")
    print("="*91)    
    print(f"Time in the air (experimental): {time_in_air_exp:.4f} s")
    print(f"Time in the air (theoretical): {time_in_air_theory:.4f} s")
    print(f"Discrepancy in time: {time_discrepancy:.4f} s ({time_discrepancy_percentage:.2f}%)")

    print("-"*91)
    
    print(f"Average angular velocity: {omega_exp}")
    print(f"Final angle (experimental): {touch_ground_orientation}")
    print(f"Final angle (theoretical): {theta_final_theory}")
    print(f"Discrepancy in final angle: {theta_final_discrepancy} ({theta_final_discrepancy_percentage}%)")
    
    print("="*91)

    return time_discrepancy, theta_final_discrepancy

def plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, block_release_ver_velocity):
    plt.figure(figsize=(8, 18))

    plt.subplot(3, 1, 1)
    plt.plot(masses, time_discrepancies, 'o-', color='blue', label='Time Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Time Discrepancy (s)')
    plt.title('Mass vs Time Discrepancy')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    angle_discrepancies = np.array(angle_discrepancies)
    plt.plot(masses, angle_discrepancies[:, 0], 'o-', color='red', label='Angle Discrepancy X')
    plt.plot(masses, angle_discrepancies[:, 1], 'o-', color='green', label='Angle Discrepancy Y')
    plt.plot(masses, angle_discrepancies[:, 2], 'o-', color='orange', label='Angle Discrepancy Z')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Angle Discrepancy (radians)')
    plt.title('Mass vs Angle Discrepancy')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(masses, block_release_ver_velocity, 'o-', color='purple', label='Block Release Vertical Velocity')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.title('Mass vs Block Release Vertical Velocity')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'discrepancy_vs_mass.png')
    plt.savefig(filename)
    plt.close()

def main(iteration=1, render_mode=None, contact_vis=None, block_mass=0.1):

    masses = []
    time_discrepancies = []
    angle_discrepancies = []
    block_release_ver_velocities = []
    local_time = datetime.now()

    for i in range(iteration):
        block_mass = round(np.random.uniform(0.050, 0.400), 3)
        print(f"Running iteration {i+1} with block mass: {block_mass}")

        output_dir = f'outputs/{local_time}'
        sub_output_dir = f'outputs/{local_time}/{i}_{block_mass:.3f}'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)

        block_positions_orientations = [([0.65, 0.2, 0], [0, 0, 0])]
        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations, block_mass=block_mass)
        
        model = mujoco.MjModel.from_xml_string(world_xml_model)
        data = mujoco.MjData(model)
        contact = mujoco.MjContact()

        mujoco.mj_kinematics(model, data)

        fsm = FiniteStateMachine(model)
        renderer = SimulationRenderer(model, data, output_dir=sub_output_dir, render_mode=render_mode, contact_vis=contact_vis)

        state = 'initial_pose'
        has_block_steady = False

        block_trans_vel_preflip_hist = []
        block_position_hist = []
        block_orientation_hist = []
        block_trans_vel_hist = []
        block_ang_vel_hist = []
        time_hist = []
        qvel_hist = []

        trigger_iteration = 0
        screenshot_iteration = 0


        while has_block_steady == False and data.time < 6:
            mujoco.mj_step(model, data)
            #qvel_hist.append(data.qvel.tolist())
            #data.qfrc_applied[17] = 0.01

            block_position, block_orientation = get_block_pose(model, data, 'block_0')
            block_trans_velocity, block_ang_velocity = get_block_velocity(data)
            time = data.time
            #time_hist.append(time)

            # Initial Pose
            if state not in ['flip_block', 'move_back']:
                current_position, _ = get_ee_pose(model, data)
                state = fsm.reset_pose(model, data, current_position)
                if state == 'approach_block':
                    block_trans_vel_preflip_hist.append(block_trans_velocity.tolist())
            
            # Flip block
            else:
                block_position_hist.append(block_position.tolist())
                block_orientation_hist.append(block_orientation.tolist())
                block_trans_vel_hist.append(block_trans_velocity.tolist())
                block_ang_vel_hist.append(block_ang_velocity.tolist())
                time_hist.append(time)

                if fsm.has_gripper_opened == False:
                    fsm.flip_block(model, data, time)
                else:
                    if fsm.state == 'post_flip_block':
                        fsm.move_back(model, data, time)
                        if screenshot_iteration == 0:
                            renderer.take_screenshot(time)
                            screenshot_iteration += 1
                    else:
                        fsm.flip_block(model, data, time)

                    # To log the release state of the block
                    if trigger_iteration == 0:
                        release_time = np.copy(time)
                        block_release_pos = np.copy(block_position)
                        block_release_orientation = np.copy(block_orientation)

                        block_release_transvel = np.copy(block_trans_velocity)
                        block_release_angvel = np.copy(block_ang_velocity)

                        block_release_ver_velocity = block_release_transvel[2]
                        time_flight_prediction = 2 * block_release_ver_velocity / 9.81
                        landing_time_pred = release_time + time_flight_prediction
                        renderer.take_screenshot(time)
                        trigger_iteration += 1

                    # To log the state of the block when touch the floor for the first time
                    if has_block_landed(data, block_position) == True:
                        if trigger_iteration == 1:
                            touch_ground_time = np.copy(time)
                            renderer.take_screenshot(time)
                            
                            closest_index = np.argmin(np.abs(np.array(time_hist) - touch_ground_time))
                            block_touch_ground_position = block_position_hist[closest_index]
                            block_touch_ground_orientation = block_orientation_hist[closest_index]
                            
                            trigger_iteration += 1

                    # To log the state of the block when already landed steadily
                    if np.linalg.norm(block_trans_velocity) < 0.01 and np.linalg.norm(block_ang_velocity) < 0.01:
                        if trigger_iteration == 2:
                            has_block_steady = True
                            steady_time = np.copy(time)
                            
                            closest_index = np.argmin(np.abs(np.array(time_hist) - steady_time))
                            block_steady_position = block_position_hist[closest_index]
                            block_steady_orientation = block_orientation_hist[closest_index]
                            
                            trigger_iteration += 1


            renderer.render_frame(time)

        print("="*91)
        print("Iteration: ", i)
        print("Block release time: ", release_time)
        print("Release EE velocity: ", fsm.release_ee_velocity) 
        print (f"Block release position : {block_release_pos} Block release orientation: {block_release_orientation}")
        print(f"Block translational release velocity: {block_release_transvel} Block angular release velocity: {block_release_transvel}")

        print("="*91)

        print(f"Block touch the ground time: {touch_ground_time}")

        print("Position when the block touch the ground: ", block_touch_ground_position) 
        print("Orientation when the block touch the ground: ", block_touch_ground_orientation)
        print("Position when the block landed steadily: ", block_steady_position) 
        print("Orientation when the block landed steadily: ", block_steady_orientation)

        
        plot_joint_velocities(sub_output_dir, release_time, fsm.time_hist, fsm.joint_vel_hist, fsm.target_joint_vel_hist)
        plot_block_pose(sub_output_dir, release_time, landing_time_pred, touch_ground_time, steady_time,  time_hist, block_position_hist, block_orientation_hist, block_trans_vel_hist, block_ang_vel_hist)
        plot_velocity_comparison(sub_output_dir, release_time, time_hist, fsm.ee_vel_hist, block_trans_vel_hist)
        time_discrepancy, angle_discrepancy = check_physical_assumptions(
            release_time=release_time,  
            touch_ground_time=touch_ground_time, 
            block_release_ver_velocity=block_release_ver_velocity, 
            block_release_orientation=block_release_orientation, 
            touch_ground_orientation=block_touch_ground_orientation,
            time_hist = time_hist,
            block_ang_vel_hist = block_ang_vel_hist
        )

        masses.append(block_mass)
        time_discrepancies.append(time_discrepancy)
        angle_discrepancies.append(angle_discrepancy)
        block_release_ver_velocities.append(block_release_ver_velocity)

        renderer.close()

    plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, block_release_ver_velocities)


if __name__ == "__main__":
    main(iteration=30, render_mode='video_side')

