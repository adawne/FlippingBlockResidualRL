import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import argparse
import inspect

import matplotlib.pyplot as plt
from datetime import datetime

from sim_utils import *
from mujoco_utils import *
from scene_builder import *
from finite_state_machine import *


def main(iteration, render_modes, contact_vis, random_mass, block_mass, block_size, ee_flip_target_velocity, sim_description, 
        block_solimp, block_solref, block_friction, cone, noslip_iterations, noslip_tolerance, impratio, pad_friction, 
        pad_solimp, pad_solref, clampness, use_random_parameters):
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
                                      noslip_tolerance, impratio, pad_friction, pad_solimp, pad_solref, clampness, use_random_parameters)
        output_dir, sub_output_dir = create_directories_and_save_config(i, block_mass, formatted_time, render_modes, current_config)

        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations, 
                                        block_mass=block_mass, block_size=block_size, block_solimp=block_solimp, 
                                        block_solref=block_solref, block_friction=block_friction, cone=cone, 
                                        noslip_iterations=noslip_iterations, noslip_tolerance=noslip_tolerance,impratio=impratio, 
                                        pad_friction=pad_friction, pad_solimp=pad_solimp, pad_solref=pad_solref)
        model = mujoco.MjModel.from_xml_string(world_xml_model)
        #model.opt.timestep = 0.008
        print(f"Simulation timestep: {model.opt.timestep}")
        data = mujoco.MjData(model)
        contact = mujoco.MjContact()
        mujoco.mj_resetData(model, data)
        fsm = FiniteStateMachine(model)
        if use_random_parameters is not True:
            renderer = SimulationRenderer(model, data, output_dir=sub_output_dir, local_time=formatted_time, render_modes=render_modes, contact_vis=contact_vis)

        state = 'initial_pose'
        has_block_steady = False

        block_trans_vel_preflip_hist = []
        block_position_hist = []
        block_orientation_hist = []
        block_trans_vel_hist = []
        block_ang_vel_hist = []
        time_hist = []
        qvel_hist = []
        contact_hist = []

        mpc_joint_hist = []

        trigger_iteration = 0
        screenshot_iteration = 0
        save_csv_iteration = 0
        debug_iteration = 0

        release_time = None
        touch_ground_time = None
        block_touch_ground_position = None
        block_touch_ground_orientation = None
        block_touch_ground_height = None
        block_touch_ground_velocity = None

        while has_block_steady == False and data.time < 8:
            #print(data.contact.geom1, data.contact.geom2)
            time = data.time
            mujoco.mj_step(model, data)
                
            if use_random_parameters is not True:
                renderer.render_frame(time)

            if fsm.simulation_stop:
                renderer.close()
                break

            # Contact analysis
            if fsm.state == 'flip_block':
                contact_hist.append(analyze_contacts(data, model, fsm, time))

            block_position, block_orientation = get_block_pose(model, data, quat=True)
            block_trans_velocity = data.sensor('block_linvel').data.copy()
            block_ang_velocity = data.sensor('block_angvel').data.copy()
            #time_hist.append(time)
                

            # Initial pose
            if state not in ['flip_block', 'move_back']:
                current_position, _ = get_ee_pose(model, data)
                state = fsm.reset_pose(model, data, time, current_position)
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
                    #fsm.flip_block(model, data, time, ee_flip_target_velocity)
                    fsm.flip_block_mpc(model, data, time)
#===========================================================================================================

                # After flipping block
                else:
                    # When no contact at all
                    if has_block_released(data) and trigger_iteration == 0:
                        release_time = time
                        block_release_pos, block_release_orientation = get_block_pose(model, data, quat=True)
                        _, block_release_euler = get_block_pose(model, data)
                        block_release_transvel = data.sensor('block_linvel').data.copy()
                        block_release_angvel = data.sensor('block_angvel').data.copy()
                        renderer.take_screenshot(time)
                            
                        trigger_iteration += 1 
                    # Moving back
                    if fsm.state == 'post_flip_block':
                        fsm.move_back(model, data, time)
                        if screenshot_iteration == 0: 
                            if use_random_parameters is not True:
                                renderer.take_screenshot(time)
                            screenshot_iteration += 1
                    # Holding position
                    else:
                        #fsm.flip_block(model, data, time, ee_flip_target_velocity)
                        fsm.flip_block_mpc(model, data, time)

                    # To log the release state of the block
                    if trigger_iteration == 1:
                        if use_random_parameters is not True:
                            renderer.take_screenshot(time)
                        trigger_iteration += 1

                    # To log the state of the block when touch the floor for the first time
                    if has_block_landed(data) == True:
                        if trigger_iteration == 2:
                            touch_ground_time = np.copy(time)
                            if use_random_parameters is not True:
                                renderer.take_screenshot(time)
                            
                            closest_index = np.argmin(np.abs(np.array(time_hist) - touch_ground_time))
                            block_touch_ground_position = block_position_hist[closest_index]
                            block_touch_ground_height = block_touch_ground_position[2]
                            block_touch_ground_orientation = block_orientation_hist[closest_index]
                            block_touch_ground_velocity = block_trans_velocity
                            
                            trigger_iteration += 1

                    # To log the state of the block when already landed steadily
                    if np.linalg.norm(block_trans_velocity) < 0.001 and np.linalg.norm(block_ang_velocity) < 0.001:
                        if trigger_iteration == 3:
                            has_block_steady = True
                            steady_time = np.copy(time)
                            if use_random_parameters is not True:
                                renderer.take_screenshot(time)
                            
                            closest_index = np.argmin(np.abs(np.array(time_hist) - steady_time))
                            block_steady_position = block_position_hist[closest_index]
                            block_steady_orientation = block_orientation_hist[closest_index]
                            
                            trigger_iteration += 1


        if use_random_parameters is not True:
            source_lines, line_number = inspect.getsourcelines(log_simulation_results)
            print(f"Location of logsim results: {inspect.getfile(log_simulation_results)} | {line_number}")
            log_simulation_results(
                i=i,
                release_time=release_time,
                fsm=fsm,
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
                    "fsm_release_ee_velocity": fsm.release_ee_linvel.tolist(), "has_block_steady": has_block_steady}
            return iteration_result
            
        else:
            steady_time = 0
            block_steady_position = [0, 0, 0] 
            block_steady_orientation = [0, 0, 0, 0] 
            print("Block has not landed steadily, using default values for position and orientation.")


    # if use_random_parameters is not True:
    #     print("SAVE SIM STATS TRIGGERED")
    #     save_sim_stats(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocities)
    #     plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocities)


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
    parser.add_argument('--use_random_parameters', action='store_true', help="Use parameter combinations")
    
    args = parser.parse_args()
    
    if args.use_random_parameters:
        parameter_combinations = generate_simulation_parameters()
        all_results = []
        
        i = 0
        for params in parameter_combinations:
            block_mass, block_friction, cone, noslip_iterations, \
            noslip_tolerance, impratio, pad_friction, pad_solimp, pad_solref, clampness = params
            print("===Iteration: ", i," ====")

            result = main(iteration= args.iteration, render_modes=args.render_modes, contact_vis=args.contact_vis, 
                 random_mass=args.random_mass, block_mass=block_mass, block_size=args.block_size, 
                 ee_flip_target_velocity=args.ee_flip_target_velocity, sim_description=args.sim_description,
                 block_solimp=[0.99, 0.995, 0.0000001, 0.5, 2], 
                 block_solref=[0.005, 2], 
                 block_friction=block_friction, cone=cone, noslip_iterations=noslip_iterations, 
                 noslip_tolerance=noslip_tolerance, impratio=impratio, pad_friction=pad_friction, 
                 pad_solimp=pad_solimp, pad_solref=pad_solref, clampness=clampness)
            
            if result is not None:
                all_results.append(result)
            i += 1
        
        output_file = os.path.join('outputs/comb_test', 'all_simulation_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
    
    else:
        main(**vars(args))