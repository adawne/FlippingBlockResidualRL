import os
import numpy as np
#from research_main.envs.utils import *
import matplotlib.pyplot as plt

from ur_kinematics import *
from mujoco_utils import *
from sim_utils import *


class FiniteStateMachine:
    def __init__(self, model):
        self.state = 'initial_pose'
        self.move_to_next_state = True
        self.has_block_released = False
        self.iteration = 0
        
        self.shoulder_pan_id = model.joint('shoulder_pan_joint').id
        self.shoulder_lift_id = model.joint('shoulder_lift_joint').id
        self.elbow_id = model.joint('elbow_joint').id
        self.wrist_1_id = model.joint('wrist_1_joint').id
        self.wrist_2_id = model.joint('wrist_2_joint').id
        self.wrist_3_id = model.joint('wrist_3_joint').id

        self.active_motors_list = [self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id]
        self.active_motors = ActuatorController(self.active_motors_list)
        self.active_motors.switch_to_position_controller(model)

        self.has_block_grasped = False
        self.has_block_flipped = False
        self.passive_motor_angles_hold = None

        self.blue_subbox_to_left_pad_contact = False
        self.blue_subbox_to_right_pad_contact = False

        self.trigger_iteration = 0

        self.current_ee_pose =0

        self.time_hist = []
        self.joint_vel_hist = []
        self.target_joint_vel_hist = []
        self.ee_vel_hist = []


    def reset_pose(self, model, data, time, current_position):
        self.has_block_released = False

        if self.state == 'initial_pose':
            self._initial_pose(model, data, current_position)
        elif self.state == 'approach_block':
            self._transition_to_approach_block(data, current_position)
        elif self.state == 'grasp_block':
            self._transition_to_grasp_block(model, data, time, current_position)
        elif self.state == 'prep_flip_block':
            self._prepare_for_flip(model, data, current_position)
        elif self.state == 'final_prep_flip_block':
            self._final_prepare_for_flip(model, data, current_position)
        elif self.state == 'post_flip_block':
            self.state = 'initial_pose'

        if self.state != 'final_prep_flip_block':
            move_ee_to_point(model, data, self.target_position, self.target_orientation)

        return self.state
            
            
    def _initial_pose(self, model, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.915, 0.2, 0.4]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        #print(np.linalg.norm(np.subtract(current_position, self.target_position)))
        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.025:
            self.move_to_next_state = True
            self.state = 'approach_block'

    def _transition_to_approach_block(self, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.915, 0.2, 0.30]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        #print(np.linalg.norm(np.subtract(current_position, self.target_position)))
        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.02:
            self.move_to_next_state = True
            self.state = 'grasp_block'

    def _transition_to_grasp_block(self, model, data, time, current_position):
        self.move_to_next_state = False
        self.iteration += 1
        gripper_close(data)

        for j in range(data.ncon):
            geom1_name = model.geom(data.contact[j].geom1).name
            geom2_name = model.geom(data.contact[j].geom2).name
            distance = data.contact[j].dist

            if ((geom1_name == 'blue_subbox' and geom2_name == 'left_pad1') or
                (geom1_name == 'left_pad1' and geom2_name == 'blue_subbox')):
                if 0 >= distance >= -0.0005:
                    self.blue_subbox_to_left_pad_contact = True

            if ((geom1_name == 'blue_subbox' and geom2_name == 'right_pad1') or
                (geom1_name == 'right_pad1' and geom2_name == 'blue_subbox')):
                if 0 >= distance >= -0.0005:
                    self.blue_subbox_to_right_pad_contact = True


        if self.blue_subbox_to_left_pad_contact and self.blue_subbox_to_right_pad_contact:
        #if self.iteration > 500:
            self.grasping_time = time
            self.has_block_grasped = True
            self.move_to_next_state = True
            self.iteration = 0
            self.state = 'prep_flip_block'

    def _prepare_for_flip(self, model, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.915, 0.2, 0.34]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.025:
            self.move_to_next_state = True
            self.state = 'final_prep_flip_block'

    def _final_prepare_for_flip(self, model, data, current_position):
        self.move_to_next_state = False
        target_joint_angles = [-np.pi/2, -1.0763, 1.6462, -2*np.pi/3, -np.pi/2, 0]
        set_joint_states(data, self.active_motors_list, target_joint_angles)

        if np.linalg.norm(np.subtract(get_joint_angles(data), target_joint_angles)) < 0.02:
            self.move_to_next_state = True

            self.active_motors_list = [self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id]
            self.passive_motors_list = []
            self.active_motors = ActuatorController(self.active_motors_list)
            self.passive_motors = ActuatorController(self.passive_motors_list)

            self.active_motors.switch_to_velocity_controller(model)
            self.passive_motors.switch_to_velocity_controller(model)
            #print(get_joint_angles(data))

            self.state = 'flip_block'

    def flip_block(self, model, data, time, ee_flip_target_velocity):
        self.time_hist.append(time)
        
        joint_velocities = np.array(get_joint_velocities(data)).flatten()
        self.joint_vel_hist.append(joint_velocities)
        
        self.prev_ee_pose = self.current_ee_pose
        self.current_ee_pose = np.copy(get_ee_pose(model, data)[0])
        self.ee_velocity = get_ee_velocity(self.prev_ee_pose, self.current_ee_pose)
        self.ee_vel_hist.append(self.ee_velocity)

        if self.has_block_flipped == False:
            #gripper_close(data, clampness)
            target_trans_velocities = ee_flip_target_velocity
            target_ang_velocities = [0, -np.pi, 0]
            target_joint_velocities = calculate_joint_velocities_trans_partial(model, data, target_trans_velocities, target_ang_velocities)
            #target_joint_velocities = calculate_joint_velocities_full(model, data, target_trans_velocities, target_ang_velocities)
            #print(target_joint_velocities)
            #print(self.ee_velocity)
            self.target_joint_vel_hist.append(target_joint_velocities.copy())

            block_position, block_orientation = get_block_pose(model, data, 'block_0')

            shoulder_lift_vel_bias = -0.125
            elbow_vel_bias = -0.075
            biased_target_joint_velocities = target_joint_velocities.copy()
            biased_target_joint_velocities[1] += shoulder_lift_vel_bias
            biased_target_joint_velocities[2] += elbow_vel_bias
            biased_target_joint_velocities[3] = -np.pi
            set_joint_states(data, self.active_motors_list, biased_target_joint_velocities)

            
            if block_orientation[1] < -1.042:
            #if get_specific_joint_angles(data, [self.wrist_1_id])[0]  > - 2.0944:
                gripper_open(data)
                print("Release wrist 1 angle: ", get_specific_joint_angles(data, [self.wrist_1_id])[0])
                self.has_block_flipped = True
                self.release_time = time
                self.release_ee_velocity = self.ee_velocity 

                self.active_motors_list = []
                self.passive_motors_list = [self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id]
                self.active_motors = ActuatorController(self.active_motors_list)
                self.passive_motors = ActuatorController(self.passive_motors_list)

                self.passive_motors.switch_to_position_controller(model)
                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)       

        else:
            self.target_joint_vel_hist.append(np.zeros((6)))

            self.trigger_iteration += 1
            set_joint_states(data, self.passive_motors_list, self.passive_motor_angles_hold)

            if self.trigger_iteration > 50:
                #self.state = 'post_flip_block'
                self.trigger_iteration = 0
                self.active_motors_list = [self.wrist_1_id]
                self.passive_motors_list = [self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_2_id, self.wrist_3_id]
                self.active_motors = ActuatorController(self.active_motors_list)
                self.passive_motors = ActuatorController(self.passive_motors_list)

                self.active_motors.switch_to_position_controller(model)
                self.passive_motors.switch_to_position_controller(model)
                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)       

    def move_back(self, model, data, time):
        self.prev_ee_pose = self.current_ee_pose
        self.current_ee_pose = np.copy(get_ee_pose(model, data)[0])
        self.ee_velocity = get_ee_velocity(self.prev_ee_pose, self.current_ee_pose)
        self.ee_vel_hist.append(self.ee_velocity)
        set_joint_states(data, self.active_motors_list, [np.pi])

        #if get_specific_joint_angles(data, [self.shoulder_lift_id])[0]  > - 2.9:
        #    target_trans_velocities = [-1, 0, 0]
        #    target_orientation = [0, -np.pi/2, 0]
        #    target_joint_velocities = calculate_joint_velocities(model, data, target_trans_velocities, target_orientation)
        #    self.target_joint_vel_hist.append(target_joint_velocities.copy())
        
            # Apply bias only for some joints
        #    shoulder_lift_vel_bias = 0
        #    elbow_vel_bias = -0.075

        #    biased_target_joint_velocities = target_joint_velocities.copy()
        #    biased_target_joint_velocities[1] += shoulder_lift_vel_bias
        #    biased_target_joint_velocities[2] += elbow_vel_bias

        #    self.time_hist.append(time)
        #    joint_velocities = np.array(get_joint_velocities(data)).flatten()
        #    self.joint_vel_hist.append(joint_velocities)

        #    set_joint_states(data, self.active_motors_list, biased_target_joint_velocities)
        #else:
        #    if self.trigger_iteration == 0:
        #        self.active_motors_list = []
        #        self.passive_motors_list = [self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id]
        #        self.active_motors = ActuatorController(self.active_motors_list)
        #        self.passive_motors = ActuatorController(self.passive_motors_list)

        #        self.passive_motors.switch_to_position_controller(model)
        #        self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)
        #        self.trigger_iteration += 1
        #    else:
        #        set_joint_states(data, self.passive_motors_list, self.passive_motor_angles_hold)