import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2

import matplotlib.pyplot as plt

from utils import *
from scene_builder import *
from finite_state_machine import *

def compute_reward_test(data, block_orientation_hist):
    'To DO: Fix detect block contact function, Make a function to detect if block fall behind the robot'
    #if has_hit_robot(state):
    #    reward = -150
    if has_rotated(block_orientation_hist) and has_flipped(data):
        reward = 100
    elif has_flipped(data) and not has_rotated(block_orientation_hist):
        reward = -50
    elif has_rotated(block_orientation_hist) and not has_flipped(data):
        reward = 30
    else:
        reward = -100
    
    return reward

def main(render_mode=None, contact_vis=None):
    block_positions_orientations = [([0.5, 0.5, 0.1], [0, 0, 0])]
    world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations)
    
    model = mujoco.MjModel.from_xml_string(world_xml_model)
    data = mujoco.MjData(model)
    contact = mujoco.MjContact()

    mujoco.mj_kinematics(model, data)

    fsm = FiniteStateMachine(model)
    
    if render_mode is not None:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        
        if render_mode == 'livecam':
            viewer = mujoco.viewer.launch_passive(model, data)
        else:
            renderer = mujoco.Renderer(model, height=1024, width=1440)
            framerate = 60
            frames = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('random_test.mp4', fourcc, framerate, (1440, 1024))

    if contact_vis is not None:
        options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(options)
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        # tweak scales of contact visualization elements
        model.vis.scale.contactwidth = 0.1
        model.vis.scale.contactheight = 0.03
        model.vis.scale.forcewidth = 0.05
        model.vis.map.force = 0.3

    state = 'initial_pose'
    has_block_landed = False

    block_position_hist = []
    block_orientation_hist = []
    block_trans_vel_hist = []
    block_ang_vel_hist = []
    time_hist = []
    
    print_iteration = 0

    while data.time < 6:
    #while has_block_landed == False:
        # Reset pose
        if state not in ['flip_block', 'move_back']:
            current_position, _ = get_ee_pose(model, data)
            state = fsm.reset_pose(model, data, current_position)
        
        # Flip block
        else:
            release_elbow_angle = 0.9
            if get_specific_joint_angles(data, [model.joint('elbow_joint').id])[0] < release_elbow_angle:
                action = 1
            else:
                action = 0
            fsm.do_flip(model, data, action)

            if fsm.has_block_released:
                block_position, block_orientation = get_block_pose(model, data, 'block_0')
                block_trans_velocity, block_ang_velocity = get_block_velocity(data)
                
                block_position_hist.append(block_position.tolist())
                block_orientation_hist.append(block_orientation.tolist())
                block_trans_vel_hist.append(block_trans_velocity.tolist())
                block_ang_vel_hist.append(block_ang_velocity.tolist())
                time_hist.append(data.time)

                # To log the release state of the block
                if print_iteration == 0:
                    release_time = data.time
                    release_block_pos = get_block_pose(model, data, 'block_0')[0]
                    #print("Lift pos: ", fsm.lift_block_pos)
                    #print(release_block_pos[2], fsm.lift_block_pos[2], release_time, fsm.lift_time)
                    block_init_ver_velocity = block_trans_velocity[2]
                    time_flight_prediction = 2 * block_init_ver_velocity / 9.81
                    
                    print("Release time: ", release_time, "Block position :", release_block_pos, "Release vertical velocity: ", block_init_ver_velocity)
                    print_iteration += 1

                if block_position[2] < 1e-1:
                    if print_iteration == 1:
                        print("The block touch the ground: ", data.time)
                        landing_time_pred = release_time + time_flight_prediction
                        closest_index = np.argmin(np.abs(np.array(time_hist) - landing_time_pred))
                        position_at_predicted_landing = block_position_hist[closest_index]
                        print("Landing time prediction: ", landing_time_pred, "Position at landing time prediction: ", position_at_predicted_landing)
                        print_iteration += 1
                        #has_block_landed = True


                if np.linalg.norm(block_trans_velocity) < 0.01 and np.linalg.norm(block_ang_velocity) < 0.001:
                    #print("Print iteration: ", print_iteration)
                    has_block_landed = True
                    

                    #if print_iteration == 2:
                    #    landing_time = data.time
                    #    print("Landing time: ", landing_time) 
                    #    print("Time in the air: ", landing_time-release_time) 
                    #    print_iteration += 1 

            #terminated = has_block_landed

            
        mujoco.mj_step(model, data)

        if render_mode is not None:
            if render_mode == 'livecam':
                viewer.sync()
            else:
                if len(frames) < data.time * framerate:
                    if render_mode == 'video_upper':
                        camera.distance = 3
                        camera.azimuth = 190
                        camera.elevation = -45
                    elif render_mode == 'video_side':
                        camera.distance = 3
                        camera.azimuth = 130
                        camera.elevation = -25
                    if contact_vis is not None:
                        renderer.update_scene(data, camera, options)
                    else:
                        renderer.update_scene(data, camera)
                    pixels = renderer.render()
                    frames.append(pixels)
                    out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    #print("Length of time history array: ", len(time_hist), "Length of block pos history array: ", len(block_position_hist), "Length of block trans vel array: ",len(block_trans_vel_hist), "Length of block ang vel history array: ", len(block_ang_vel_hist))
    # After the simulation loop ends and before the plotting starts
    print(landing_time_pred)
    landing_time_line = landing_time_pred if has_block_landed else None

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
    if landing_time_line:
        plt.axvline(x=landing_time_line, color='r', linestyle='--', label='Landing time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Block Position History')
    plt.legend()

    # Plotting block orientation history
    plt.subplot(4, 1, 2)
    plt.plot(time_hist, block_orientation_hist[0], label='X orientation')
    plt.plot(time_hist, block_orientation_hist[1], label='Y orientation')
    plt.plot(time_hist, block_orientation_hist[2], label='Z orientation')
    if landing_time_line:
        plt.axvline(x=landing_time_line, color='r', linestyle='--', label='Landing time')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation')
    plt.title('Block Orientation History')
    plt.legend()

    # Plotting translational velocity qvel[14:17]
    plt.subplot(4, 1, 3)
    plt.plot(time_hist, block_trans_vel_hist[0], label='Vx')
    plt.plot(time_hist, block_trans_vel_hist[1], label='Vy')
    plt.plot(time_hist, block_trans_vel_hist[2], label='Vz')
    if landing_time_line:
        plt.axvline(x=landing_time_line, color='r', linestyle='--', label='Landing time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Translational Velocity qvel[14:17]')
    plt.legend()

    # Plotting angular velocity qvel[17:20]
    plt.subplot(4, 1, 4)
    plt.plot(time_hist, block_ang_vel_hist[0], label='Omega x')
    plt.plot(time_hist, block_ang_vel_hist[1], label='Omega y')
    plt.plot(time_hist, block_ang_vel_hist[2], label='Omega z')
    if landing_time_line:
        plt.axvline(x=landing_time_line, color='r', linestyle='--', label='Landing time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity qvel[17:20]')
    plt.legend()

    plt.tight_layout()
    plt.savefig('block_release_velocities_and_positions_lighter.png')

    #if has_rotated(block_orientation_hist) and has_flipped(data):
    #    reward = 100
    #elif has_flipped(data) and not has_rotated(block_orientation_hist):
    #    reward = -50
    #elif has_rotated(block_orientation_hist) and not has_flipped(data):
    #    reward = 30
    #else:
    #    reward = -100

    #print(get_block_pose(model, data, 'block_0'))
    #print("Reward: ", reward,  has_rotated(block_orientation_hist), has_flipped(data))
    if render_mode != 'livecam':
        out.release()


#========================================================================


if __name__ == "__main__":
    main(render_mode='video_upper')