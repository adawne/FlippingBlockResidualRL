import numpy as np
import time
import pybullet as pb
import cvxpy as cp

from utils import *
from torque_controller import *

# Define states
INITIAL_STATE = 0 
INITIAL_GRASP_BLOCK = 1
LIFTING_BLOCK = 2
ASSEMBLY = 3
RELEASE_BLOCK = 4
REVERSE_TRAJECTORY = 5
GRASP_BLOCK = 6 
CALIBRATE = 7
POST_ASSEMBLY = 8
APPROACH_GRASP_BLOCK = 9
APPROACH_RELEASE = 10
FINISH = 11

# Initialize the current state
current_state = INITIAL_STATE


class FSMFinal:
    def __init__(self):
        self.current_state = INITIAL_STATE
        self.iteration = 0
        self.is_block_assembled = False
        self.is_state_done = False

    def update(self, pb_client, robot_id, joint_angles_target, joint_velocities_target, joint_accelerations_target, current_level, side):
        self.assembled = False
        self.error_checker = 0
        
        if self.current_state == INITIAL_STATE:
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                self.iteration = 0
                self.is_state_done = True
                self.current_state = INITIAL_GRASP_BLOCK

        elif self.current_state == INITIAL_GRASP_BLOCK:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)  
                self.iteration = 0
                self.is_state_done = True
                self.current_state = LIFTING_BLOCK

        elif self.current_state == LIFTING_BLOCK:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)  
                self.iteration = 0
                self.is_state_done = True
                #if side < 3:
                self.current_state = ASSEMBLY
                #else:
                #self.current_state = CALIBRATE

        elif self.current_state == CALIBRATE:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)  
                self.iteration = 0
                self.is_state_done = True
                self.current_state = ASSEMBLY


        elif self.current_state == ASSEMBLY:
            gripper_close(pb_client, robot_id)
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            print("Joint torques: ", joint_torques)
            print("Shape of joint torques: ", joint_torques.shape)
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                self.iteration = 0
                self.is_state_done = True
                #if side < 2:
                self.current_state = RELEASE_BLOCK
                #else:
                #    release_block(pb_client, robot_id)
                #    self.current_state = APPROACH_RELEASE

        elif self.current_state == APPROACH_RELEASE:
            gripper_close(pb_client, robot_id)
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                self.iteration = 0
                self.is_state_done = True
                self.current_state = RELEASE_BLOCK

        elif self.current_state == RELEASE_BLOCK:
            gripper_close(pb_client, robot_id)
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                release_block(pb_client, robot_id)
                self.is_block_assembled = True
                self.iteration = 0
                self.is_state_done = True
                if side == 3:
                    self.current_state = POST_ASSEMBLY
                else:
                    self.current_state = REVERSE_TRAJECTORY

                #if side < 3:
                    #self.current_state = REVERSE_TRAJECTORY
                
        elif self.current_state == POST_ASSEMBLY:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)  
                self.iteration = 0
                self.is_state_done = True
                self.current_state = FINISH


        elif self.current_state == REVERSE_TRAJECTORY:
            self.is_state_done = False
            self.is_block_assembled = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                self.iteration = 0
                self.is_state_done = True
                #if side > 1: 
                    #self.current_state = APPROACH_GRASP_BLOCK
                #else:
                self.current_state = GRASP_BLOCK

        elif self.current_state == APPROACH_GRASP_BLOCK:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)
                self.iteration = 0
                self.is_state_done = True
                self.current_state = GRASP_BLOCK


        elif self.current_state == GRASP_BLOCK:
            self.is_state_done = False
            trajectory_length = len(joint_angles_target)
            joint_angles = np.array(get_joint_angles(pb_client, robot_id))
            joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))

            K_1, K_2 = compute_control_gain('accurate')
            joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[self.iteration], joint_velocities, joint_velocities_target[self.iteration], joint_accelerations_target[self.iteration])
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

            self.iteration += 1
            
            if self.iteration == trajectory_length:
                grasp_block(pb_client, robot_id)
                self.iteration = 0
                self.is_state_done = True
                self.current_state = LIFTING_BLOCK

                


        return self.current_state, self.iteration, self.is_state_done, self.is_block_assembled 