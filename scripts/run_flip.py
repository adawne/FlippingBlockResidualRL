import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import argparse

import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.RL_eval_scripts.eval_td3 import evaluate_td3
from research_main.envs.sim_utils import *
from research_main.envs.mujoco_utils import *
from research_main.envs.scene_builder import *
from flip_controller import *


def main(iteration, render_modes, contact_vis, random_mass, block_mass, block_size, ee_flip_target_velocity, sim_description, 
        block_solimp, block_solref, block_friction, cone, noslip_iterations, noslip_tolerance, impratio, pad_friction, 
        pad_solimp, pad_solref, clampness, use_mode, policy_version, policy_type):
    masses = []
    time_discrepancies = []
    angle_discrepancies = []
    height_discrepancies = []
    landing_velocities_discrepancies = []
    block_release_ver_velocities = []
    local_time = datetime.now()
    formatted_time = local_time.strftime('%Y-%m-%d_%H-%M')

    for i in range(iteration):
        if random_mass: 
            block_mass = round(np.random.uniform(0.050, 0.400),  3)
        block_positions_orientations = [([0.2, 0.2, 0], [0, 0, 0])] 
        args.block_mass = block_mass

        current_config = build_config(i, render_modes, contact_vis, random_mass, block_mass, block_size, ee_flip_target_velocity, 
                                      sim_description, block_solimp, block_solref, block_friction, cone, noslip_iterations, 
                                      noslip_tolerance, impratio, pad_friction, pad_solimp, pad_solref, clampness)
        output_dir, sub_output_dir = create_directories_and_save_config(i, block_mass, formatted_time, render_modes, current_config)


        if use_mode == "MPC_eval":
            traj_ctrl = []
            with open('../trajectories/precomputed_mpc_trajectory/interpolated_trajectory.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    ctrl_values = [float(x) for x in row['QPos'].split(',')]  
                    traj_ctrl.append(ctrl_values[:6])  

            traj_ctrl = np.array(traj_ctrl)

        elif use_mode == "RL_eval":
            evaluate_td3(policy_version=policy_version, policy_type=policy_type)
            trajectory_path = f"../trajectories/RL_trajectory/{policy_version}/raw_{policy_type}_trajectory.csv"
            traj_ctrl = []
            with open(trajectory_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    ctrl_values = [float(row[f"Ctrl_{i}"]) for i in range(6)] 
                    traj_ctrl.append(ctrl_values)
            
            traj_ctrl = np.array(traj_ctrl)



        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations, 
                                        block_mass=block_mass, block_size=block_size, block_solimp=block_solimp, 
                                        block_solref=block_solref, block_friction=block_friction, cone=cone, 
                                        noslip_iterations=noslip_iterations, noslip_tolerance=noslip_tolerance,impratio=impratio, 
                                        pad_friction=pad_friction, pad_solimp=pad_solimp, pad_solref=pad_solref, xml_mode=use_mode)
        model = mujoco.MjModel.from_xml_string(world_xml_model)
        print(f"Simulation timestep: {model.opt.timestep}")
        data = mujoco.MjData(model)
        contact = mujoco.MjContact()
        #mujoco.mj_resetData(model, data)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        controller = Controller(model, data, use_mode=use_mode)
        renderer = SimulationRenderer(model, data, output_dir=sub_output_dir, local_time=formatted_time, render_modes=render_modes, contact_vis=contact_vis)

        has_block_steady = False

        block_position_hist = []
        block_orientation_hist = []
        block_trans_vel_hist = []
        block_ang_vel_hist = []
        time_hist = []

        trigger_iteration = 0

        release_time = None
        touch_ground_time = None
        block_touch_ground_position = None
        block_touch_ground_orientation = None
        block_touch_ground_height = None
        block_touch_ground_velocity = None
        frameskip = 1

        #while data.time < 8:
        while has_block_steady == False and data.time < 8:
            time = data.time
            if use_mode == "manual_flip":
                controller.flip_block(model, data, time, ee_flip_target_velocity)
            elif use_mode == "RL_eval":
                controller.execute_flip_trajectory(model, data, time, traj_ctrl, frameskip, policy_version, policy_type)
            mujoco.mj_step(model, data, nstep = frameskip)
                
            renderer.render_frame(time)

            if controller.simulation_stop:
                renderer.close()
                break

            block_position, block_orientation = get_block_pose(model, data, quat=True)
            block_trans_velocity = data.sensor('block_linvel').data.copy()
            block_ang_velocity = data.sensor('block_angvel').data.copy()

            block_position_hist.append(block_position.tolist())
            block_orientation_hist.append(block_orientation.tolist())
            block_trans_vel_hist.append(block_trans_velocity.tolist())
            block_ang_vel_hist.append(block_ang_velocity.tolist())
            time_hist.append(time)

            if has_block_released(data) and trigger_iteration == 0:
                release_time = time
                block_release_pos, block_release_orientation = get_block_pose(model, data, quat=True)
                _, block_release_euler = get_block_pose(model, data)
                block_release_transvel = data.sensor('block_linvel').data.copy()
                block_release_angvel = data.sensor('block_angvel').data.copy()
                renderer.take_screenshot(time)
                            
                trigger_iteration += 1 
                   
            if has_block_landed(data) == True:
                if trigger_iteration == 1:
                    touch_ground_time = np.copy(time)
                    renderer.take_screenshot(time)
                    
                    closest_index = np.argmin(np.abs(np.array(time_hist) - touch_ground_time))
                    block_touch_ground_position = block_position_hist[closest_index]
                    block_touch_ground_height = block_touch_ground_position[2]
                    block_touch_ground_orientation = block_orientation_hist[closest_index]
                    block_touch_ground_velocity = block_trans_velocity
                    
                    trigger_iteration += 1

            if np.linalg.norm(block_trans_velocity) < 0.001 and np.linalg.norm(block_ang_velocity) < 0.001:
                if trigger_iteration == 2:
                    has_block_steady = True
                    steady_time = np.copy(time)
                    renderer.take_screenshot(time)
                    
                    closest_index = np.argmin(np.abs(np.array(time_hist) - steady_time))
                    block_steady_position = block_position_hist[closest_index]
                    block_steady_orientation = block_orientation_hist[closest_index]
                    
                    trigger_iteration += 1


        log_simulation_results(
            i=i,
            release_time=release_time,
            controller=controller,
            block_release_pos=block_release_pos,
            block_release_orientation=block_release_euler,
            block_release_transvel=block_release_transvel,
            block_release_angvel=block_release_angvel,
            touch_ground_time=touch_ground_time,
            block_touch_ground_position=block_touch_ground_position,
            block_touch_ground_orientation=block_touch_ground_orientation
        )
        

        time_discrepancy_percentage, angle_discrepancy_percentage, height_discrepancy_percentage, landing_velocity_discrepancy_percentage = perform_discrepancy_analysis(
            release_time=release_time, 
            touch_ground_time=touch_ground_time, 
            block_release_pos=block_release_pos, 
            block_release_transvel=block_release_transvel, 
            block_touch_ground_height=block_touch_ground_height,
            block_release_quat=block_release_orientation,  
            block_touch_ground_quat=block_touch_ground_orientation, 
            block_touch_ground_velocity=block_touch_ground_velocity, 
            time_hist=time_hist, 
            block_position_hist=block_position_hist, 
            block_ang_vel_hist=block_ang_vel_hist
        )


        # #plot_and_save_contacts(sub_output_dir, contact_hist)
        # plot_and_save_results(sub_output_dir, i, release_time, time_hist, fsm, block_trans_vel_hist, landing_time_pred, 
        #                     touch_ground_time, steady_time, block_position_hist, block_orientation_hist, 
        #                     block_ang_vel_hist, block_release_pos, block_release_orientation, block_release_transvel, 
        #                     block_release_angvel, block_touch_ground_position, block_touch_ground_orientation,
        #                     block_steady_position, block_steady_orientation)

        masses.append(block_mass)
        time_discrepancies.append(time_discrepancy_percentage)
        angle_discrepancies.append(angle_discrepancy_percentage)
        height_discrepancies.append(height_discrepancy_percentage)
        landing_velocities_discrepancies.append(landing_velocity_discrepancy_percentage)
        block_release_ver_velocities.append(block_release_transvel[2])

        renderer.close()


        if has_block_steady:
            print("Position when the block landed steadily: ", block_steady_position) 
            print("Orientation when the block landed steadily: ", block_steady_orientation)
            iteration_result = {"current_config": current_config, "block_release_transvel": block_release_transvel.tolist(), 
                    "fsm_release_ee_velocity": controller.release_ee_linvel.tolist(), "has_block_steady": has_block_steady}
            return iteration_result
            
        else:
            steady_time = 0
            block_steady_position = [0, 0, 0] 
            block_steady_orientation = [0, 0, 0, 0] 
            print("Block has not landed steadily, using default values for position and orientation.")


    save_sim_stats(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocities)
    plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with custom configuration.")
    
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--render_modes', nargs='+', default=['livecam'])
    parser.add_argument('--contact_vis', type=bool, default=False)
    parser.add_argument('--random_mass', type=bool, default=False)
    parser.add_argument('--block_mass', type=float, default=0.1)
    parser.add_argument('--block_size', nargs='+', type=float, default=[0.03, 0.02, 0.08])
    parser.add_argument('--ee_flip_target_velocity', nargs='+', type=float, default=[0.1, 0, 0.75])
    parser.add_argument('--sim_description', type=str, default='default_sim')

    parser.add_argument('--block_solimp', nargs='+', type=float, default=[0.99, 0.995, 0.0000001, 0.5, 2])
    parser.add_argument('--block_solref', nargs='+', type=float, default=[0.005, 2])
    parser.add_argument('--block_friction', nargs='+', type=float, default=[5, 0.01, 0.001])
    parser.add_argument('--cone', type=str, default='pyramidal')
    parser.add_argument('--noslip_iterations', type=int, default=5)
    parser.add_argument('--noslip_tolerance', type=float, default=1e-6)
    parser.add_argument('--impratio', type=int, default=1)
    parser.add_argument('--pad_friction', type=int, default=5)
    parser.add_argument('--pad_solimp', nargs='+', type=float, default=[0.97, 0.99, 0.001])
    parser.add_argument('--pad_solref', nargs='+', type=float, default=[0.004, 1])
    parser.add_argument('--clampness', type=int, default=220)
    parser.add_argument('--use_mode', type=str, default="manual_flip")
    parser.add_argument('--policy_version', type=str, default="TD3 v0.1.1 -noMPC -QVel -FixStart", help="Policy version folder name")
    parser.add_argument('--policy_type', type=str, default="final_policy", choices=["final_policy", "best_policy"],
                        help="Type of policy to use (final_policy or best_policy)")
    args = parser.parse_args()
    
    main(**vars(args))