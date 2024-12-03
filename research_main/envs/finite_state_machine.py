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
        self.has_gripper_opened = False
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
        self.mpc_qpos_log = []
        self.mpc_qvel_log = []
        self.mpc_ctrl_log = []
        self.time_plot_log = []
        self.qpos_plot_log = []
        self.qvel_plot_log = []

        self.time_hist = []
        self.joint_vel_hist = []
        self.target_joint_vel_hist = []
        self.ee_linvel_hist = []
        self.ee_angvel_hist = []

        self.simulation_stop = False

        self.mpc_ctrl = []
        with open('precomputed_mpc_traj/interpolated_trajectory.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ctrl_values = [float(x) for x in row['QPos'].split(',')]  
                self.mpc_ctrl.append(ctrl_values[:6])  # Only the first 6 qpos values

        self.mpc_ctrl = np.array(self.mpc_ctrl)

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
            #print(f"Pinch pos: {data.sensor('pinch_pos').data.copy()} | Pinch quat: {data.sensor('pinch_quat').data.copy()}")
            #print(f"QPos: {data.qpos.copy()} | QVel: {data.qvel.copy()}")
            init_mpc_qpos = self.mpc_ctrl[self.mpc_timestep]
            data.ctrl[:6] = init_mpc_qpos
            self.update_motors_controller(model,
                                    active_ids=[self.shoulder_pan_id, self.shoulder_lift_id, self.elbow_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id],
                                    passive_ids=[],
                                    mode="position")
            print(f"Qpos: {data.qpos}")
            print(f"Qvel: {data.qvel}")
            print(f"Ctrl: {data.ctrl}")
            print(f"Block height: {get_block_pose(model, data)[0]}")
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
        
        ee_linvel = data.sensor('pinch_linvel').data.copy()
        ee_angvel = data.sensor('pinch_angvel').data.copy()
        self.ee_linvel_hist.append(ee_linvel)
        self.ee_angvel_hist.append(ee_angvel)


        if self.has_gripper_opened == False:
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
                self.has_gripper_opened = True
                self.release_time = time
                self.release_ee_linvel = ee_linvel
                self.release_ee_angvel = ee_angvel


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

    def flip_block_mpc(self, model, data, time, frameskip=1):

        _, block_orientation = get_block_pose(model, data)
        quat_desired_block = R.from_euler('xyz', [0, np.pi-2.0945, -3.14]).as_quat()
        quat_current_block = np.array(data.sensor('block_quat').data.copy()) 
        #quat_desired = np.array([-0.0001253, -0.31613082, -0.00038637, -0.94871552])

        quat_current_ee = np.array(data.sensor('pinch_quat').data.copy())
        quat_dist = quat_distance(quat_current_block, quat_desired_block)

        #print(f"Quat offset: {quat_offset} | Quat distance: {quat_dist}")
        
        self.time_hist.append(time)
        joint_velocities = np.array(get_joint_velocities(data)).flatten()
        self.joint_vel_hist.append(joint_velocities)
        ee_linvel = data.sensor('pinch_linvel').data.copy()
        ee_angvel = data.sensor('pinch_angvel').data.copy()
        self.ee_linvel_hist.append(ee_linvel)
        self.ee_angvel_hist.append(ee_angvel)
        
        relative_quat = R.from_quat(quat_current_block) * R.from_quat(quat_current_ee).inv()
        #print(relative_quat.as_euler('xyz', degrees=True))  # Relative orientation in degrees

        if self.mpc_timestep <= 105:
        #if block_orientation[1] > -1.042:
        #if self.mpc_timestep < len(self.mpc_ctrl) and quat_dist > 5e-3:
            given_ctrl = self.mpc_ctrl[self.mpc_timestep]
            #data.ctrl[[0, 4, 5]] = [-1.58, -np.pi/2, 0]
            #data.ctrl[[1, 2, 3]] = [-2 * np.pi, -3.01, -2.06]
            #data.ctrl[:6] = given_ctrl
            print(self.mpc_timestep, data.qpos[[1, 2, 3]].copy())

            # Log data at each time step
            if not hasattr(self, 'csv_initialized'):
                with open('mpc_log.csv', 'w', newline='') as csvfile:
                    log_writer = csv.writer(csvfile)
                    log_writer.writerow(['Time', 'Ctrl', 'QPos', 'QVel'])  # Write headers
                self.csv_initialized = True  # Prevent rewriting headers

            with open('mpc_log.csv', 'a', newline='') as csvfile:
                log_writer = csv.writer(csvfile)
                ctrl_str = ','.join(map(str, data.ctrl[:6].copy()))
                qpos_str = ','.join(map(str, data.qpos[:6].copy()))
                qvel_str = ','.join(map(str, data.qvel[:6].copy()))
                log_writer.writerow([time, ctrl_str, qpos_str, qvel_str])

            self.mpc_timestep += frameskip

        else:
            gripper_open(data)
            self.has_gripper_opened = True
            self.release_time = time
            self.release_ee_linvel = ee_linvel
            self.release_ee_angvel = ee_angvel 


        # if self.has_gripper_opened == False:
        #     # if block_orientation[1] > -1.04:
        #     # #if data.qpos[3] < -4.5:
        #     #     #print(self.mpc_timestep)
        #     #     given_ctrl = self.mpc_ctrl_log[self.mpc_timestep]
        #     #     #set_joint_states(data, [self.wrist_1_id], [-4.1544])
        #     #     set_joint_states(data, self.active_motors_list, given_ctrl)

        #     # else:
        #     #     gripper_open(data)
        #     #     print("Release wrist 1 angle: ", wrist_1_angle)
        #     #     self.has_gripper_opened = True
        #     #     self.release_time = time
        #     #     self.release_ee_velocity = ee_velocity 
        #     #     self.motor_angles_hold = get_specific_joint_angles(data, self.active_motors_list)
        #     #     #print("Motor holds: ", self.motor_angles_hold)
                


    def move_back(self, model, data, time):
        self.prev_ee_pose = self.current_ee_pose
        self.current_ee_pose = np.copy(get_ee_pose(model, data)[0])
        self.ee_linvel = data.sensor('pinch_linvel').data.copy()
        self.ee_linvel_hist.append(self.ee_linvel)
        set_joint_states(data, self.active_motors_list, [np.pi])

    # def move_back_mpc(self):
    #     print("TRIGGERED")

