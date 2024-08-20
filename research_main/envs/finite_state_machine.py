import numpy as np
#from research_main.envs.utils import *

from utils import *

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

        self.has_gripper_opened = False
        self.passive_motor_angles_hold = None
        self.print_iteration = 0 


    def reset_pose(self, model, data, current_position):
        self.has_block_released = False

        if self.state == 'initial_pose':
            self._initial_pose(current_position)
        elif self.state == 'approach_block':
            self._transition_to_approach_block(current_position)
        elif self.state == 'grasp_block':
            self._transition_to_grasp_block(data, current_position)
        elif self.state == 'lift_block':
            self._transition_to_lift_block(current_position)
        elif self.state == 'prep_flip_block':
            self._prepare_for_flip(model, data)
        elif self.state == 'post_flip_block':
            self.state = 'initial_pose'

        if self.state not in ['prep_flip_block']:
            move_ee_to_point(model, data, self.target_position, self.target_orientation)

        return self.state

    def _initial_pose(self, current_position):
        self.move_to_next_state = False
        self.target_position = [0.65, 0.2, 0.4]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
            self.move_to_next_state = True
            self.state = 'approach_block'

    def _transition_to_approach_block(self, current_position):
        self.move_to_next_state = False
        self.target_position = [0.65, 0.2, 0.28]
        self.target_orientation = [np.pi / 2, -np.pi, 0]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
            self.move_to_next_state = True
            self.state = 'grasp_block'

    def _transition_to_grasp_block(self, data, current_position):
        self.move_to_next_state = False
        self.iteration += 1
        gripper_close(data)

        if self.iteration > 1000:
            self.move_to_next_state = True
            self.iteration = 0
            self.state = 'lift_block'

    def _transition_to_lift_block(self, current_position):
        self.move_to_next_state = False
        self.iteration += 1
        self.target_position = [0.65, 0.2, 0.7]

        if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.02:
            self.move_to_next_state = True
            self.iteration = 0
            self.state = 'prep_flip_block'

    def _prepare_for_flip(self, model, data):
        # Reassign actuator groups for the flip stage
        self.active_motors_list = [self.shoulder_lift_id, self.elbow_id]
        self.passive_motors_list = [self.shoulder_pan_id, self.wrist_1_id, self.wrist_2_id, self.wrist_3_id]
        self.active_motors = ActuatorController(self.active_motors_list)
        self.passive_motors = ActuatorController(self.passive_motors_list)

        # Set control modes
        self.active_motors.switch_to_velocity_controller(model)
        self.passive_motors.switch_to_position_controller(model)

        target_velocities = [2*np.pi/3, -4*np.pi/6]
        stop_elbow_angle = [1.35]

        set_joint_states(data, self.active_motors_list, target_velocities)

        if np.linalg.norm(np.subtract(get_specific_joint_angles(data, [self.elbow_id]), stop_elbow_angle)) < 0.02:
            self.move_to_next_state = True
            self.iteration = 0

            self.active_motors_list = [self.shoulder_lift_id, self.elbow_id, self.wrist_1_id]
            self.passive_motors_list = [self.shoulder_pan_id, self.wrist_2_id, self.wrist_3_id]
            self.active_motors = ActuatorController(self.active_motors_list)
            self.passive_motors = ActuatorController(self.passive_motors_list)

            self.active_motors.switch_to_velocity_controller(model)
            self.passive_motors.switch_to_position_controller(model)
            self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)

            self.state = 'flip_block'

    def do_flip(self, model, data, action):
        release = action

        if self.state == 'flip_block':
            self.move_to_next_state = False

            target_velocities = [np.pi/4, -np.pi/3, -np.pi] #Worked: -np.pi/2, -np.pi (but not rotated)
            stop_velocities = [0, 0, 0]

            set_joint_states(data, self.passive_motors_list, self.passive_motor_angles_hold)
            set_joint_states(data, self.active_motors_list, target_velocities)
            
            if release == 1:
                set_joint_states(data, self.active_motors_list, stop_velocities)
                gripper_open(data)
                self.has_gripper_opened = True
                self.has_block_released = True

    def do_move_back(self, model, data):
        self.iteration += 1
        if self.iteration > 50 and self.state == 'flip_block':       
            self.move_to_next_state = True
            self.iteration = 0
            self.state = 'move_back'

        if self.state == 'move_back':
            self.move_to_next_state = False

            moving_velocity_actuators = [self.elbow_id, self.wrist_1_id]
            moving_position_actuators = [self.shoulder_pan_id]

            target_velocity = [np.pi, np.pi]
            target_position = [0]
            
            stop_elbow_angle = 2.4
            current_elbow_angles = get_specific_joint_angles(data, [self.elbow_id])


            set_joint_states(data, moving_velocity_actuators, target_velocity)
            set_joint_states(data, moving_position_actuators, target_position)

            if (np.linalg.norm(np.subtract(get_specific_joint_angles(data, [self.elbow_id]), stop_elbow_angle)) < 0.02):
                self.move_to_next_state = True
                self.iteration = 0
                self.state = 'post_flip_block'
                set_joint_states(data, moving_velocity_actuators, [0, 0])

        