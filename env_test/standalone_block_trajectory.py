import sys
import os
import mujoco
import mujoco.viewer
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'research_main', 'envs'))
sys.path.append(parent_dir)

from sim_utils import *
from scene_builder import *
from mujoco_utils import *
from scipy.spatial.transform import Rotation as R
from env_test_utils import *

def make_block_float(model, data):
    block_body_id = model.body('block_0').id
    block_mass = model.body('block_0').mass[0]
    gravity = -model.opt.gravity[2]
    data.qfrc_applied[16] = block_mass * gravity
    #data.xfrc_applied[block_body_id][2] = block_mass*gravity

def main(render_modes):
    local_time = datetime.now()
    formatted_time = local_time.strftime('%Y-%m-%d_%H-%M')
    output_dir, sub_output_dir = create_directories(i=0, block_mass=0.1, formatted_time=formatted_time, render_modes=render_modes)

    block_positions_orientations = [([0.2, 0.2, 0.35], [0, 0, 0])]
    world_xml_model = create_ur_model(use_mode="debug", 
                                    block_positions_orientations=block_positions_orientations)
    
    model = mujoco.MjModel.from_xml_string(world_xml_model)
    print(f"Simulation timestep: {model.opt.timestep}")
    data = mujoco.MjData(model)
    contact = mujoco.MjContact()
    mujoco.mj_resetData(model, data)
    renderer = SimulationRenderer(model, data, output_dir=sub_output_dir, local_time=formatted_time, render_modes=render_modes)
    
    block_body_id = model.body('block_0').id
    make_block_float(model, data)

    times = []
    block_positions = []
    block_orientations_quat = []
    block_orientations_euler = []
    desired_orientations_quat = []
    block_trans_vels = []
    block_ang_vels = []
    xfrc_applied_data = []
    qfrc_applied_data = []

    desired_angle_quat_array = R.from_euler('zyx', [0, 2.0944, 0]).as_quat()
    prev_orientation_y = None

    flipped = False
    trigger_iteration = 0

    while data.time < 3:
        time = data.time
        mujoco.mj_step(model, data)
        renderer.render_frame(time)
        block_trans_velocity = data.sensor('block_linvel').data.copy()
        block_ang_velocity = data.sensor('block_angvel').data.copy()
        block_position, block_orientation = get_block_pose(model, data, quat=True)

        rotation = R.from_quat(block_orientation)
        block_orientation_euler = rotation.as_euler('zyx', degrees=True)


        if flipped:
            if has_block_landed(data) == True:
                if trigger_iteration == 1:
                    touch_ground_time = np.copy(time)
                    renderer.take_screenshot(time)
                    
                    closest_index = np.argmin(np.abs(np.array(times) - touch_ground_time))
                    block_touch_ground_position = block_positions[closest_index]
                    block_touch_ground_height = block_touch_ground_position[2]
                    block_touch_ground_euler = block_orientations_euler[closest_index]
                    block_touch_ground_quat = block_orientations_quat[closest_index]
                    block_touch_ground_velocity = block_trans_velocity
                    
                    trigger_iteration += 1


        else:
            make_block_float(model, data)
            v_x_desired = 0.1
            v_y_desired = 0.0111
            omega_y_desired = -3.8

            if block_orientation_euler[1] > -60:
                if prev_orientation_y is not None and block_orientation_euler[1] > prev_orientation_y:
                    data.xfrc_applied[block_body_id] = np.zeros(6) 
                    data.qfrc_applied[:] = 0
                    flipped = True
                    if trigger_iteration == 0:
                        flipped_time = np.copy(time)
                        block_release_pos = np.copy(block_position)
                        block_release_euler = np.copy(block_orientation_euler)
                        block_release_quat = np.copy(block_orientation)
                        block_release_transvel = np.copy(block_trans_velocity)
                        block_release_angvel = np.copy(block_ang_velocity)
                        block_release_ver_velocity = block_release_transvel[2]
                        
                        renderer.take_screenshot(time)
                        trigger_iteration += 1

            if block_trans_velocity[0] < v_x_desired:
                data.xfrc_applied[block_body_id][0] = 0.02

            if block_trans_velocity[2] < v_y_desired:
                data.xfrc_applied[block_body_id][2] = 0.01
                #data.qfrc_applied[16] = 0.01

            if block_ang_velocity[1] > omega_y_desired:
                data.xfrc_applied[block_body_id][4] = -0.02
                #data.qfrc_applied[18] = 0.01

        times.append(time)
        block_orientations_quat.append(block_orientation)
        block_orientations_euler.append(block_orientation_euler)  
        desired_orientations_quat.append(desired_angle_quat_array)
        xfrc_applied_data.append(data.xfrc_applied[block_body_id].copy())
        qfrc_applied_data.append(data.qfrc_applied[14:20].copy())
        block_positions.append(block_position.tolist())
        block_trans_vels.append(block_trans_velocity.tolist())
        block_ang_vels.append(block_ang_velocity.tolist())
        
        prev_orientation_y = block_orientation_euler[1]

    log_simulation_results(flipped_time, block_release_pos, block_release_euler, 
                            block_release_transvel, block_release_angvel, touch_ground_time, 
                            block_touch_ground_position, block_touch_ground_euler)
    plot_block_data(times, block_orientations_quat, desired_orientations_quat, block_orientations_euler, 
                    xfrc_applied_data, qfrc_applied_data, block_positions, block_trans_vels, block_ang_vels, 
                    flipped_time, touch_ground_time)


    time_discrepancy_percentage, angle_discrepancy_percentage, height_discrepancy_percentage, landing_velocity_discrepancy_percentage = perform_discrepancy_analysis(
        release_time=flipped_time, 
        touch_ground_time=touch_ground_time, 
        block_release_pos=block_release_pos, 
        block_release_transvel=block_release_transvel, 
        block_touch_ground_height=block_touch_ground_height,
        block_release_quat=block_release_quat,  
        block_touch_ground_quat=block_touch_ground_quat, 
        block_touch_ground_velocity=block_touch_ground_velocity, 
        time_hist=times, 
        block_position_hist=block_positions, 
        block_ang_vel_hist=block_ang_vels
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with custom configuration.")    
    parser.add_argument('--render_modes', nargs='+', default=['livecam'])

    args = parser.parse_args()

    main(render_modes= args.render_modes)
