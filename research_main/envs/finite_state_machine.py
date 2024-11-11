import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from ur_kinematics import *
from mujoco_utils import *
from sim_utils import *


class FiniteStateMachine:
    def __init__(self, model):
        self.state = 'initial_pose'
        self.move_to_next_state = True
        self.has_block_released = False
        
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
        self.current_ee_pose = 0
        self.mpc_timestep = 0
        self.mpc_debug_print = 0
        self.save_mpc_csv = True

        self.time_log = []
        self.qpos_log = []
        self.qvel_log = []

        self.mpc_time_log = []
        self.mpc_state_log = []
        self.mpc_state_log_2 = []
        self.mpc_qvel_log = []

        self.time_hist = []
        self.joint_vel_hist = []
        self.target_joint_vel_hist = []
        self.ee_vel_hist = []

        self.simulation_stop = False

        self.mpc_ctrl_log = []
        with open('precomputed_mpc_traj/interpolated_trajectory.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ctrl_values = [float(x) for x in row['Ctrl'].split(',')]  
                self.mpc_ctrl_log.append(ctrl_values[:6])  # Only the first 6 qpos values

        self.mpc_ctrl_log = np.array(self.mpc_ctrl_log)

    def update_motors_controller(self, model, active_ids, passive_ids, mode="position"):
        # Update the lists of active and passive motors
        self.active_motors_list = active_ids
        self.passive_motors_list = passive_ids

        # Reinitialize ActuatorControllers with new lists
        self.active_motors = ActuatorController(self.active_motors_list)
        self.passive_motors = ActuatorController(self.passive_motors_list)

        # Switch the controllers based on the mode
        if mode == "position":
            self.active_motors.switch_to_position_controller(model)
            self.passive_motors.switch_to_position_controller(model)
        elif mode == "velocity":
            self.active_motors.switch_to_velocity_controller(model)
            self.passive_motors.switch_to_velocity_controller(model)


    def reset_pose(self, model, data, time, current_position):
        self.has_block_released = False

        if self.state == 'initial_pose':
            self._initial_pose(model, data, current_position)
        elif self.state == 'approach_block':
            self._transition_to_approach_block(data, current_position)
        elif self.state == 'grasp_block':
            self._transition_to_grasp_block(model, data, time, current_position)
        elif self.state == 'lift_block':
            self._lift_block(model, data, current_position)
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
        self.target_position = [0.2, 0.2, 0.4]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        #print(np.linalg.norm(np.subtract(current_position, self.target_position)))
        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.025:
            self.move_to_next_state = True
            self.state = 'approach_block'

    def _transition_to_approach_block(self, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.2, 0.2, 0.28]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        #print(np.linalg.norm(np.subtract(current_position, self.target_position)))
        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.02:
            self.move_to_next_state = True
            self.state = 'grasp_block'

    def _transition_to_grasp_block(self, model, data, time, current_position):
        self.move_to_next_state = False
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
            #self.iteration = 0
            self.state = 'lift_block'

    def _lift_block(self, model, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.2, 0.2, 0.30]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.025:
            self.trigger_iteration += 1
            if self.trigger_iteration > 100:
                self.move_to_next_state = True
                self.trigger_iteration = 0
                self.state = 'prep_flip_block'

    def _prepare_for_flip(self, model, data, current_position):
        self.move_to_next_state = False
        self.target_position = [0.2, 0.2, 0.30]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.025:
            self.move_to_next_state = True
            self.final_flip_joint_angles = np.copy(get_joint_angles(data))
            self.state = 'final_prep_flip_block'

    def _final_prepare_for_flip(self, model, data, current_position):
        self.move_to_next_state = False
        self.final_flip_joint_angles = [-1.5923697502906031,-2.2355866024308724,2.702949141598393,-2.067142791964366,-1.572676790518878,-4.223297354370639e-09]                                                                                                               
  
        # self.final_flip_joint_angles[0] = -np.pi/2
        # self.final_flip_joint_angles[-1] = 0
        # self.final_flip_joint_angles[-2] = -np.pi/2
        set_joint_states(data, self.active_motors_list, self.final_flip_joint_angles)

        if np.linalg.norm(np.subtract(get_joint_angles(data), self.final_flip_joint_angles)) < 0.02:
            self.move_to_next_state = True
            self.update_motors_controller(model,
                                    active_ids=[self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id],
                                    passive_ids=[],
                                    mode="velocity")
            self.state = 'flip_block'
            #print("Qpos before flip: ", get_joint_angles(data))
            # with open("flip_data_log.txt", "a") as file:
            #     file.write(f"Qpos: {data.qpos.tolist()}\n")
            #     file.write(f"Ctrl: {data.ctrl.tolist()}\n")
            #     file.write("\n")  # Add a blank line for readability


    def flip_block(self, model, data, time, ee_flip_target_velocity):
        self.time_hist.append(time)
        
        joint_velocities = np.array(get_joint_velocities(data)).flatten()
        self.joint_vel_hist.append(joint_velocities)
        
        ee_velocity = get_ee_velocity(model, data)[0]
        self.ee_vel_hist.append(ee_velocity)

        #print(get_joint_angles(data))
        # self.time_log.append(data.time)
        # self.qpos_log.append(get_joint_angles(data))
        # self.qvel_log.append(get_joint_velocities(data))

        if self.has_block_flipped == False:
            #gripper_close(data, clampness)
            target_trans_velocities = ee_flip_target_velocity
            target_ang_velocities = [0, -np.pi, 0]
            target_joint_velocities = calculate_joint_velocities_trans_partial(model, data, target_trans_velocities, target_ang_velocities)
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
                self.release_ee_velocity = ee_velocity 

                self.update_motors_controller(model,
                                        active_ids=[],
                                        passive_ids=[self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id],
                                        mode="position"
                                        )
                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)       
    

        else:
            self.target_joint_vel_hist.append(np.zeros((6)))

            self.trigger_iteration += 1
            set_joint_states(data, self.passive_motors_list, self.passive_motor_angles_hold)

            if self.trigger_iteration > 50:
                self.state = 'post_flip_block'
                self.trigger_iteration = 0
                self.update_motors_controller(model,
                                        active_ids=[self.wrist_1_id],
                                        passive_ids=[self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_2_id, self.wrist_3_id],
                                        mode="position")

                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)           
                #self.save_trajectory_to_csv_and_plot()

    def flip_block_mpc(self, model, data, time):

        ee_velocity = get_ee_velocity(model, data)[0]
        wrist_1_angle = get_specific_joint_angles(data, [self.wrist_1_id])[0]
        
        if self.has_block_flipped == False:
            if wrist_1_angle > -4.1544:
                #print(self.mpc_timestep)
                given_ctrl = self.mpc_ctrl_log[self.mpc_timestep]
                #set_joint_states(data, [self.wrist_1_id], [-4.1544])
                set_joint_states(data, self.active_motors_list, given_ctrl)

            else:
                gripper_open(data)
                print("Release wrist 1 angle: ", wrist_1_angle)
                self.has_block_flipped = True
                self.release_time = time
                self.release_ee_velocity = ee_velocity 
                self.motor_angles_hold = get_specific_joint_angles(data, self.active_motors_list)
                #print("Motor holds: ", self.motor_angles_hold)

                print("Framepos: ", data.sensor('pinch_pos').data.copy())
                print("Framelinvel: ", data.sensor('pinch_linvel').data.copy())
                print("Frameangvel: ", data.sensor('pinch_angvel').data.copy())

        else:
            set_joint_states(data, self.active_motors_list, self.motor_angles_hold)

            # if self.trigger_iteration > 50:
            #     self.state = 'post_flip_block'
            #     self.trigger_iteration = 0
            #     self.update_motors_controller(model,
            #                             active_ids=[self.wrist_1_id],
            #                             passive_ids=[self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_2_id, self.wrist_3_id],
            #                             mode="position")

            #     self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list) 

        self.mpc_timestep += 1
 

    def move_back(self, model, data, time):
        self.prev_ee_pose = self.current_ee_pose
        self.current_ee_pose = np.copy(get_ee_pose(model, data)[0])
        self.ee_velocity = get_ee_velocity(model, data)[0]
        self.ee_vel_hist.append(self.ee_velocity)
        set_joint_states(data, self.active_motors_list, [np.pi])

    def move_back_mpc(self):
        print("TRIGGERED")

