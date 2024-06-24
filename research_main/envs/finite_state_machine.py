import numpy as np
from utils_push_mujoco import *

class FiniteStateMachine:
    def __init__(self):
        self.state = 'move_to_target'
        self.move_to_next_state = True
        self.iteration = 0
    
    def update(self, data, current_position):
        if self.state == 'move_to_target':
            self.move_to_next_state = False
            self.target_position = [0.5, 0.5, 0.4]
            self.target_orientation = [0, -np.pi, 0]   

            if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
                    self.move_to_next_state = True
                    self.state = 'approach_block'
        
        elif self.state == 'approach_block':
            self.move_to_next_state = False
            self.target_position = [0.5, 0.5, 0.3]
            gripper_open(data)

            if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
                self.move_to_next_state = True
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
            self.target_position = [0.5, 0.5, 0.7]
                
            #if np.linalg.norm(np.subtract(current_position, self.target_position)) < 0.015:
                #self.move_to_next_state = True
                #self.state = 'release_block'


        elif self.state == 'release_block':
            self.iteration += 1

            if self.iteration > 1000:
                gripper_open(data)
            

        #print(self.state)
            
        return self.target_position, self.target_orientation, self.move_to_next_state, self.iteration
