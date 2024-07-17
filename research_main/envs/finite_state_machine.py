import numpy as np
#from research_main.envs.utils import *

from utils import *

class FiniteStateMachine:
    def __init__(self, model):
        self.state = 'initial_pose'
        self.move_to_next_state = True
        self.iteration = 0
        self.has_block_released = False
        
        self.shoulder_pan_id = model.joint('shoulder_pan_joint').id
        self.shoulder_lift_id = model.joint('shoulder_lift_joint').id
        self.elbow_id = model.joint('elbow_joint').id
        self.wrist_1_id = model.joint('wrist_1_joint').id
        self.wrist_2_id = model.joint('wrist_2_joint').id
        self.wrist_3_id = model.joint('wrist_3_joint').id

        self.passive_flipping_actuators = [self.shoulder_pan_id, self.shoulder_lift_id, self.wrist_2_id, self.wrist_3_id]
        self.passive_motors = ActuatorController(self.passive_flipping_actuators)
        self.passive_motors.switch_to_position_controller(model)
        self.passive_motor_angles_hold = np.zeros(len(self.passive_flipping_actuators))
        
        self.active_flipping_actuators = [self.elbow_id, self.wrist_1_id]
        self.has_gripper_opened = False

        self.print_iteration = 0 


    def reset_pose(self, model, data, current_position):
        self.has_block_released = False
        self.active_motors = ActuatorController(self.active_flipping_actuators)
        self.active_motors.switch_to_position_controller(model)

        if self.state == 'initial_pose':
            self.move_to_next_state = False
            self.target_position = [0.5, 0.5, 0.4]
            self.target_orientation = [np.pi/2, -np.pi, 0]   

            if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
                    self.move_to_next_state = True
                    self.state = 'approach_block'
        
        elif self.state == 'approach_block':
            self.move_to_next_state = False
            self.target_position = [0.5, 0.5, 0.28]
            self.iteration += 1
            gripper_open(data)

            if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015 :
                self.move_to_next_state = True
                self.iteration = 0
                self.state = 'grasp_block'

        elif self.state == 'grasp_block':
            self.move_to_next_state = False
            self.iteration += 1
            gripper_close(data)

            if self.iteration > 1000:
                self.move_to_next_state = True
                self.iteration = 0
                self.state = 'lift_block'

        elif self.state == 'lift_block':
            self.move_to_next_state = False
            self.iteration += 1
            self.target_position = [0.5, 0.5, 0.7]
                 
            if (np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.02):
                self.move_to_next_state = True
                self.iteration = 0
                self.state = 'pre_flip_block'


        elif self.state == 'pre_flip_block':
            self.move_to_next_state = False
            self.iteration += 1
            moving_actuators = [self.shoulder_lift_id, self.wrist_3_id]
            target_angles = [-1.25, 0]

            set_joint_states(data, moving_actuators, target_angles)
            #print(np.linalg.norm(np.subtract(get_specific_joint_angles(data, moving_actuators), target_angles)))

            if (np.linalg.norm(np.subtract(get_specific_joint_angles(data, moving_actuators), target_angles)) < 0.02):
                self.move_to_next_state = True
                self.iteration = 0
                #print("Prepare to flip block: ", data.time, get_block_pose(model, data, 'block_0')[1])
                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_flipping_actuators)
                self.state = 'flip_block'
                #print("Prepare to flip block: ", get_specific_joint_angles(data, self.active_flipping_actuators))

        elif self.state == 'post_flip_block':
            self.state = 'initial_pose'

        if self.state not in ['pre_flip_block']:
            move_ee_to_point(model, data, self.target_position, self.target_orientation)

        return self.state

    def do_flip(self, model, data, action):
        elbow_discrete_action = action[0]
        wrist_1_discrete_action = action[1]
        release = action[2]

        elbow_velocity = map_to_velocity(elbow_discrete_action)
        wrist_1_velocity = map_to_velocity(wrist_1_discrete_action)

        if self.state == 'flip_block':
            #if self.print_iteration == 0:
            #    print("Begin flipping block: ", get_specific_joint_angles(data, self.active_flipping_actuators))
            #    self.print_iteration += 1

            self.active_motors.switch_to_velocity_controller(model)
            
            self.move_to_next_state = False
            moving_actuators = self.active_flipping_actuators
            
            target_velocities = [elbow_velocity, wrist_1_velocity] #Worked: -np.pi/2, -np.pi (but not rotated)
            stop_velocities = [0, 0]

            set_joint_states(data, self.passive_flipping_actuators, self.passive_motor_angles_hold)
            set_joint_states(data, moving_actuators, target_velocities)
            #print("Block pose: ", get_block_pose(model, data, 'block_0')[0])


            if release == 1:
                set_joint_states(data, moving_actuators, stop_velocities)
                gripper_open(data)
                self.has_gripper_opened = True
                #self.iteration += 1
                #self.print_iteration += 1

                #if self.iteration > 50:
                    #print("Block released: ", get_block_pose(model, data, 'block_0')[0])           
                    #self.has_block_released = True
                    #self.move_to_next_state = True
                    #self.iteration = 0
                    #self.state = 'move_back'

    def do_move_back(self, model, data):
        self.iteration += 1
        if self.iteration > 50 and self.state == 'flip_block':
            print("Block released: ", get_block_pose(model, data, 'block_0')[0])           
            self.has_block_released = True
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

            #print("Joint velocities: ", get_joint_velocities(data))

            set_joint_states(data, moving_velocity_actuators, target_velocity)
            set_joint_states(data, moving_position_actuators, target_position)

            if (np.linalg.norm(np.subtract(get_specific_joint_angles(data, [self.elbow_id]), stop_elbow_angle)) < 0.02):
                self.move_to_next_state = True
                self.iteration = 0
            #    #print("Prepare to flip block: ", data.time, get_block_pose(model, data, 'block_0')[1])
            #    self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_flipping_actuators)
                self.state = 'post_flip_block'
                set_joint_states(data, moving_velocity_actuators, [0, 0])

        