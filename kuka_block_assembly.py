'''
This is a demo of kuka robot block assembly.
'''

import os, time
import numpy as np

import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc

from utils import *
from torque_controller import *
from trajectories_generator import *
from finite_state_machine import *

if __name__ == '__main__':
    ## Initialize the simulator
    pb_client = initializeGUI(enable_gui=False, gravity=-10)

    ## Load urdf models
    plane_id = pb_client.loadURDF("plane.urdf")
    pb_client.changeDynamics(plane_id,
                             -1,
                             lateralFriction=1,
                             spinningFriction=.001,
                             rollingFriction=.001,
                             restitution=0,
                             collisionMargin=0,
                             linearDamping=1,
                             angularDamping=1,
                             jointDamping=1,
                             contactStiffness=-10,
                             contactDamping=-10
                             )
    
    robot_id = pb_client.loadURDF('./robot_models/kuka_gripper.urdf',
                                  [0, 0, 0], useFixedBase=True)

    ## Draw a frame at the end effector
    draw_frame(pb_client, robot_id, 7, axis_length=.1, line_width=1, line_from=(0,0,.065))

    ## Free joints
    free_joint_torques(pb_client, robot_id, arm_joint_indices)

    ## Draw lines for prohibited area for block stacking
    pb_client.addUserDebugLine(lineFromXYZ=(0, -.5, .01),
                               lineToXYZ=(0, .5, .01),
                               lineColorRGB=(1,0,0),
                               lineWidth=3)
    pb_client.addUserDebugLine(lineFromXYZ=(0, .5, .01),
                               lineToXYZ=(1, .5, .01),
                               lineColorRGB=(1,0,0),
                               lineWidth=3)
    pb_client.addUserDebugLine(lineFromXYZ=(1, .5, .01),
                               lineToXYZ=(1, -.5, .01),
                               lineColorRGB=(1,0,0),
                               lineWidth=3)
    pb_client.addUserDebugLine(lineFromXYZ=(1, -.5, .01),
                               lineToXYZ=(0, -.5, .01),
                               lineColorRGB=(1,0,0),
                               lineWidth=3)

    ## Main loop
    is_velocity_limit_violated = False
    is_block_assembled = False

    new_block_id = -1
    floor = 0
    side = 0

    block_id = place_new_block(pb_client)
    block_target_position, block_target_orientation = update_block_target(floor, side)
    side += 1
    is_block_assembled = False


    finite_state_machine = FSMFinal()
    
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
    current_state = INITIAL_STATE
    
    current_level = 0
    current_side = 1

    joint_angles_init = np.array(get_joint_angles(pb_client, robot_id))

    joint_angles_target, joint_velocities_target, min_velocity = generate_initial_trajectory(pb_client, robot_id, end_effector_link_id, block_id, current_state, current_level, current_side)
    joint_angles_target, joint_velocities_target, joint_accelerations_target, untruncated_index = generate_final_trajectory(pb_client, robot_id, end_effector_link_id, joint_angles_target, min_velocity)


    expected_velocity = velocity_expectation(pb_client, robot_id, end_effector_link_id, joint_angles_target, joint_velocities_target)
    print("Expected velocity: ", len(expected_velocity))

    highest_velocity = 0
    lowest_velocity = 999999
    #final_level = 13
    #====================================================================================================================
    for n in range(240*600): # Do not change the number (240*600)
        #block_id = place_new_block(pb_client)
        
        if (block_id > 0):
            new_block_id = block_id

        #print("=============Waypoint index: ", n, "====================")
        is_block_assembled = False
        block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
        joint_angles = np.array(get_joint_angles(pb_client, robot_id))
        

        state, state_iteration, is_state_done, is_block_assembled = finite_state_machine.update(pb_client, robot_id, joint_angles_target, joint_velocities_target, joint_accelerations_target, current_level, current_side)
 

        if is_state_done:
            if state == FINISH:
                break
            #if state == FINISH:
                #break
            
            if state == REVERSE_TRAJECTORY:
                current_level += 1
                #if current_side == 0:
                #    current_side = 1
                #elif current_side == 1:
                #    current_side = 2
                #    current_level += 1
                #elif current_side == 2:
                #    current_side = 3
                #elif current_side == 3:
                #    current_side = 0
                #    current_level += 1

                #else:
                    #current_level +=1 
                #i#f current_level == 1 and current_side == 1: #Jumlah blok tidur
                    #break
                if current_level == 9:
                    current_side = 3  

            joint_angles_target_new, joint_velocities_target_new, min_velocity = generate_initial_trajectory(pb_client, robot_id, end_effector_link_id, block_id, state, current_level, current_side)
            
            #print("Length of new initial joint angles target: ", len(joint_angles_target_new))
            #print("Initial joint angles target before generate final trajectory: ", joint_angles_target_new[0])
            if state == LIFTING_BLOCK: 
                joint_angles_target_new, joint_velocities_target_new, joint_accelerations_target_new, untruncated_index = generate_final_trajectory(pb_client, robot_id, end_effector_link_id, joint_angles_target_new, state, min_velocity)
            else:
                joint_angles_target_new, joint_velocities_target_new, joint_accelerations_target_new, untruncated_index = generate_final_trajectory(pb_client, robot_id, end_effector_link_id, joint_angles_target_new, min_velocity, max_velocity=1.7)
            #print("Length of joint angles target: ", len(joint_angles_target))
            #print("Untruncated index: ", untruncated_index)
            #print("Joint angles real: ", joint_angles)
            
            #print("Length of new final joint angles target: ", len(joint_angles_target_new))

            joint_angles_target, joint_velocities_target, joint_accelerations_target = joint_angles_target_new, joint_velocities_target_new, joint_accelerations_target_new
            expected_velocity = velocity_expectation(pb_client, robot_id, end_effector_link_id, joint_angles_target, joint_velocities_target)
            #print("Lenght array of new expected velocity: ", len(expected_velocity))

        #print("Current state: ", state, "Is state done: ", is_state_done, "Is block assembled: ", is_block_assembled, "Current level: ", current_level, "Current side: ", current_side)


        if is_block_assembled == True:
            new_block_id = place_new_block(pb_client)
            if (new_block_id > 0):
                block_id = new_block_id
            
            new_block_target_position, new_block_target_orientation = update_block_target(floor, side)
            #side += 1
            #if side == 2:
            #    side = 0
            #    floor += 1

            block_target_position = new_block_target_position
            block_target_orientation = new_block_target_orientation

        lowest_velocity, highest_velocity = velocity_checker(pb_client, robot_id, end_effector_link_id, state_iteration, expected_velocity, highest_velocity, lowest_velocity)
        #print("Lowest velocity: ", lowest_velocity, "Highest velocity: ", highest_velocity) 
        ## ====================================================================================================================
        if (is_velocity_limit_violated):
            pb_client.addUserDebugText('Velocity Limit Violated',
                                       textPosition=(-.7,0,1.7),
                                       textSize=2,
                                       textColorRGB=(1,0,0))
            break
        
        is_velocity_limit_violated = not check_object_graspability(pb_client,
                                                                   robot_id,
                                                                   block_id)
                
        pb_client.stepSimulation()
        #time.sleep(1/240)


    ## Do not change beyond this point
    release_block(pb_client, robot_id)
    for i in range(240*60):
        print(i/240)           
        pb_client.stepSimulation()
        time.sleep(1/240)

    block_position_file = open('./block_positions_10.txt', 'w')
    for block_id in block_ids:            
        block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
        block_position_file.write('{}\n'.format(block_position))
        
        print(block_position)

    block_position_file.close()

    for _ in range(10000000000):
        pb_client.stepSimulation()
        time.sleep(1/240)

 
    
    pb.disconnect()

