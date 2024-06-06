import sys, pdb, argparse, time, pickle, itertools, json

import numpy as np
import pickle

import pybullet as pb
import pybullet_data

import gymnasium as gym
from gymnasium import spaces
import tianshou as ts

import argparse

from pybullet_utils import bullet_client

from research_main.envs.utils_push import *
from research_main.envs.torque_controller_push import *

def draw_frame(pb_client, robot_id, link_index, xyz=(0,0,0), axis_length=.2):
    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0)+np.asarray(xyz),
                               lineToXYZ=(axis_length,0,0)+np.asarray(xyz),
                               lineColorRGB=(1,0,0),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0)+np.asarray(xyz),
                               lineToXYZ=(0,axis_length,0)+np.asarray(xyz),
                               lineColorRGB=(0,1,0),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0)+np.asarray(xyz),
                               lineToXYZ=(0,0,axis_length)+np.asarray(xyz),
                               lineColorRGB=(0,0,1),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)


def grasping_control(grasping_angle):
    joint_values = (36-grasping_angle)*np.array([1, 1, -1, -1, 1, -1])

    return np.radians(joint_values)

class KukaPushBlockEnv(gym.Env):
    def __init__(self, render_mode=None):

        if render_mode is not None:
            self.pb_client = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            self.pb_client = bullet_client.BulletClient(connection_mode=pb.DIRECT)

        self.pb_client.setGravity(0, 0, -10)

        self.pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        self.pb_client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.pb_client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.pb_client.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        ## Load URDF models
        self.plane_id = self.pb_client.loadURDF("plane.urdf")
        self.robot_id = self.pb_client.loadURDF('research_main/envs/robot_models_new/kuka_gripper.urdf',
                                        basePosition=(0, 0, 0),
                                        globalScaling=1.0,
                                        useFixedBase=True)

        self.block_ids = []
        draw_frame(self.pb_client, self.plane_id, -1, xyz=(0,.5,0), axis_length=.5)

        # Observation space: joint positions (1), , joint velocities (1), block position (3), block position velocity (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float64)  # Actions for joint 0

        self.h = 1/240

    def _get_obs(self):
      
        self.joint_angles_all = np.array(get_joint_angles(self.pb_client, self.robot_id))
      
        self.joint_angles_partial = np.array([self.joint_angles_all[i] for i in [0]])
        self.joint_velocities_partial = (self.joint_angles_partial - self.prev_joint_angles_partial)/self.h
        self.prev_joint_angles_partial = self.joint_angles_partial

        self.block_position, self.block_orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=self.block_id)
        self.block_position_velocity = (np.array(self.block_position) - np.array(self.prev_block_position))/self.h
        self.prev_block_position = self.block_position

        obs = np.concatenate([self.joint_angles_partial, self.joint_velocities_partial, self.block_position, self.block_position_velocity])
        
        return obs

    def _get_info(self):
        return {
            "joint_angles_partial": self.joint_angles_partial,
            "joint_velocities_partial": self.joint_velocities_partial,
            "block_position": self.block_position,
            "block_position_velocity": self.block_position_velocity
        }

    def reset(self, seed=None, options=None):
        with open('research_main/envs/gripper_polyfit_coefficient.npy', 'rb') as f:
            poly_coefficient = np.load(f)

        gripper_control_map = np.poly1d(poly_coefficient)
            
        gripper_angle = gripper_control_map(0.10482201257840675)
        joint_values = grasping_control(gripper_angle)

        for _ in range(240):
            for i, value in enumerate(joint_values):
                self.pb_client.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                jointIndex=8+i,
                                                targetPosition=value,
                                                controlMode=pb.POSITION_CONTROL)


            self.pb_client.stepSimulation()

        for n in range(800):
            joint_values = np.zeros(7)
            position = get_end_effector_pose(self.pb_client, self.robot_id)

            for i, value in enumerate(joint_values):
                joint_angle_control(self.pb_client, self.robot_id, i, value)

            end_effector_position, end_effector_orientation = \
                get_end_effector_pose(self.pb_client, self.robot_id)

            end_effector_orientation = pb.getEulerFromQuaternion(end_effector_orientation)

            time.sleep(1/240)
            self.pb_client.stepSimulation()


        if len(self.block_ids) > 0:
            self.pb_client.removeBody(self.block_ids[-1])

        self.block_ids.append(self.pb_client.loadURDF('research_main/envs/parts/zenga_block.urdf',
                                basePosition=(.35, .4, .1),
                                baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                                # baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                                globalScaling=1,
                                useFixedBase=False))

        self.block_id = self.block_ids[-1]

        self.prev_joint_angles_all = np.array(get_joint_angles(self.pb_client, self.robot_id))
        self.prev_joint_angles_partial = np.array([self.prev_joint_angles_all[i] for i in [0]])

        self.prev_block_position, self.prev_block_orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=self.block_id)

        baseline_angle = 0.5585993153435626
        moving_direction = np.array((0-.35, .5-.4))
        starting_point = np.array([0.35555556, 0.37777778])
        starting_point -= 0.01*np.array([np.cos(baseline_angle), np.sin(baseline_angle)])
        starting_point -= .4*moving_direction

        print("Starting point: ", starting_point)
        time.sleep(1)

        for n in range(800):
            joint_values = inverse_kinematics(self.pb_client, self.robot_id, 7, (starting_point[0], starting_point[1], .32), (0, np.pi, -np.pi/2+baseline_angle))[:8]
            position = get_end_effector_pose(self.pb_client, self.robot_id)

            for i, value in enumerate(joint_values):
                joint_angle_control(self.pb_client, self.robot_id, i, value)

            end_effector_position, end_effector_orientation = \
                get_end_effector_pose(self.pb_client, self.robot_id)

            end_effector_orientation = pb.getEulerFromQuaternion(end_effector_orientation)

            self.previous_joint_values = joint_values

            time.sleep(1/240)
            self.pb_client.stepSimulation()

        observation = self._get_obs()
        info = self._get_info()


        return observation, info

    def _compute_reward(self, observation):
        target_position = np.array([0.3, 0.5])
        target_orientation = np.array([0, 0, 0])
        
        block_position = observation[2:5]

        pos_diff = np.linalg.norm(block_position[:2] - target_position[:2])

        reward = (-pos_diff)*10
        if pos_diff < 0.05:
            reward += 100  # Big reward for reaching the target

        return reward


    def step(self, action):
        print("Action: ", action)
        controlled_joints = [0]
        
        for idx, joint in enumerate(controlled_joints):
            joint_torque_control(self.pb_client, self.robot_id, joint, action[idx])
        
        for i in range(7):
            if i not in controlled_joints:
                joint_angle_control(self.pb_client, self.robot_id, i, self.previous_joint_values[i])

        time.sleep(1/240)
        self.pb_client.stepSimulation()

        observation = self._get_obs()
        reward = self._compute_reward(observation)
        info = self._get_info()
        terminated = False

        block_position = observation[2:5]
        print("Observation: ", observation)

        if np.linalg.norm(block_position[:2] - np.array([0.3, 0.5])) < 0.05:
            terminated = True
            print("Reached the target!")

        #print("Norm difference: ", np.linalg.norm(block_position[:2] - np.array([0.3, 0.5])))

        return observation, reward, terminated, False, info

    def close(self):
        pass
