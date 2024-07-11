import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2

import gymnasium as gym
from gymnasium import spaces
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
    block_positions_orientations = [([0.5, 0.5, 0.1], [np.pi/2, 0, 0])]
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
        elif render_mode == 'video':
            renderer = mujoco.Renderer(model, height=1024, width=1440)
            framerate = 60
            frames = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('gymnasium_integration_test_nocontact.mp4', fourcc, framerate, (1440, 1024))

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
    block_orientation_hist = []
    
    print_iteration = 0

    while data.time < 6:
        if state not in ['flip_block', 'move_back']:
            release_elbow_angle = 0.9
            current_position, _ = get_ee_pose(model, data)
            state = fsm.reset_pose(model, data, current_position)
        else:
            if get_specific_joint_angles(data, [model.joint('elbow_joint').id])[0] < release_elbow_angle:
                if print_iteration == 0:
                    print_iteration += 1
                action = {
                    "elbow_velocity": np.array([0]),  
                    "wrist_1_velocity": np.array([0]),  
                    "release": np.array([1]) 
                    }
            else:
                action = {
                    "elbow_velocity": np.array([-np.pi]),  
                    "wrist_1_velocity": np.array([-np.pi]),  
                    "release": np.array([0]) 
                    }

            state, has_block_released = fsm.do_flip(model, data, action)

            if has_block_released:
                _, block_orientation = get_block_pose(model, data, 'block_0')
                block_orientation_hist.append(block_orientation)
                block_trans_velocity, block_ang_velocity = get_block_velocity(data)
                if block_trans_velocity < 0.01 and block_ang_velocity < 0.001:
                    has_block_landed = True

            terminated = has_block_landed 

        mujoco.mj_step(model, data)

        if render_mode is not None:
            if render_mode == 'livecam':
                viewer.sync()
            elif render_mode == 'video':
                if len(frames) < data.time * framerate:
                    camera.distance = 4
                    camera.azimuth = 135
                    camera.elevation = -30
                    if contact_vis is not None:
                        renderer.update_scene(data, camera, options)
                    else:
                        renderer.update_scene(data, camera)
                    pixels = renderer.render()
                    frames.append(pixels)
                    out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    if has_rotated(block_orientation_hist) and has_flipped(data):
        reward = 100
    elif has_flipped(data) and not has_rotated(block_orientation_hist):
        reward = -50
    elif has_rotated(block_orientation_hist) and not has_flipped(data):
        reward = 30
    else:
        reward = -100

    print(get_block_pose(model, data, 'block_0'))
    print("Reward: ", reward,  has_rotated(block_orientation_hist), has_flipped(data))
    if render_mode == 'video':
        out.release()


#========================================================================


if __name__ == "__main__":
    main(render_mode=None)