import sys, pdb, argparse, time, pickle, itertools, json

import numpy as np

import pybullet as pb
import pybullet_data

import gymnasium as gym
import tianshou as ts

import argparse

from pybullet_utils import bullet_client

from utils_push import *
from torque_controller_push import *

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

def mark_point(pb_client, robot_id, link_index, point, axis_length=.2, line_width=1):
    point = np.array(point)

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(axis_length,0,0),
                               lineWidth=line_width,
                               lineColorRGB=(1,0,0),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,axis_length,0),
                               lineWidth=line_width,
                               lineColorRGB=(0,1,0),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,0,axis_length),
                               lineWidth=line_width,
                               lineColorRGB=(0,0,1),
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

def compute_gripper_base_pose_world(cube_base_pose, grasping_pose):
    gripper_length = .4

    cube_base_position, cube_base_orientation = cube_base_pose
    grasping_position, grasping_orientation = grasping_pose

    grasping_position_world, grasping_orientation_world = \
        pb.multiplyTransforms(positionA=cube_base_position,
                              orientationA=pb.getQuaternionFromEuler(cube_base_orientation),
                              positionB=grasping_position,
                              orientationB=pb.getQuaternionFromEuler(grasping_orientation))
    grasping_orientation_world = pb.getEulerFromQuaternion(grasping_orientation_world)

    gripper_base_position_world, gripper_base_orientation_world = \
        pb.multiplyTransforms(positionA=grasping_position_world,
                              orientationA=pb.getQuaternionFromEuler(grasping_orientation_world),
                              positionB=[0, 0, -gripper_length],
                              orientationB=pb.getQuaternionFromEuler([0, 0, 0]))
    
    gripper_base_orientation_world = pb.getEulerFromQuaternion(gripper_base_orientation_world)

    return (gripper_base_position_world, gripper_base_orientation_world)

def grasping_control(grasping_angle):
    joint_values = (36-grasping_angle)*np.array([1, 1, -1, -1, 1, -1])

    return np.radians(joint_values)

def gripper_control(current_pose, current_velocity, desired_pose):
    """
    The orientation needs to be represented in Euler angle
    """

    maximum_linear_velocity = .1 # meter per second
    force_control_gain = (-200*100, -40*100)
    torque_control_gain = (-500, -300)
    
    current_position, current_orientation = current_pose
    current_linear_velocity, current_rotational_velocity = current_velocity
    desired_position, desired_orientation = desired_pose

    current_position = np.asarray(current_position)
    current_linear_velocity = np.asarray(current_linear_velocity)
    desired_position = np.asarray(desired_position)
    
    ## Force control
    del_position = desired_position - current_position

    if (np.linalg.norm(del_position) > 0.001):
        desired_linear_velocity = maximum_linear_velocity*del_position/np.linalg.norm(del_position)

    else:
        desired_linear_velocity = 0
        
    next_gripper_position = current_position + 1/240*desired_linear_velocity
    rotation_matrix = np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(current_orientation))).reshape(3,3)

    # pdb.set_trace()
    force_control = force_control_gain[0]*(current_position-next_gripper_position) + \
        force_control_gain[1]*(current_linear_velocity-desired_linear_velocity)
    force_control = np.dot(rotation_matrix.T, force_control)
    force_control += np.dot(rotation_matrix.T, (0,0,10*10)) # gravity compensation
    
    ## Torque control
    del_gripper_orientation = pb.getDifferenceQuaternion(pb.getQuaternionFromEuler(desired_orientation),
                                                         pb.getQuaternionFromEuler(current_orientation))
    del_gripper_orientation = pb.getEulerFromQuaternion(del_gripper_orientation)
    torque_control = np.asarray(inertial_1)*(torque_control_gain[0]*np.asarray(del_gripper_orientation) + \
                                             torque_control_gain[1]*np.asarray(current_rotational_velocity))

    return force_control, torque_control

def gripper_velocity_control(current_pose, current_velocity, desired_orientation, desired_velocity):
    """
    The orientation needs to be represented in Euler angle
    """

    maximum_linear_velocity = .1 # meter per second
    force_control_gain = -500
    torque_control_gain = (-600, -300)
    
    current_position, current_orientation = current_pose

    current_linear_velocity, current_rotational_velocity = current_velocity
    current_linear_velocity = np.asarray(current_linear_velocity)
    
    desired_linear_velocity, desired_rotational_velocity = desired_velocity
    desired_linear_velocity = np.asarray(desired_linear_velocity)
    desired_rotational_velocity = np.asarray(desired_rotational_velocity)
    
    ## Force control
    rotation_matrix = np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(current_orientation))).reshape(3,3)

    # pdb.set_trace()
    force_control = force_control_gain*(current_linear_velocity-desired_linear_velocity)
    force_control = np.dot(rotation_matrix.T, force_control)
    force_control += np.dot(rotation_matrix.T, (0,0,10*10)) # gravity compensation
    
    ## Torque control
    del_gripper_orientation = pb.getDifferenceQuaternion(pb.getQuaternionFromEuler(desired_orientation),
                                                         pb.getQuaternionFromEuler(current_orientation))
    del_gripper_orientation = list(pb.getEulerFromQuaternion(del_gripper_orientation))

    torque_control = np.asarray(inertial_1)*torque_control_gain[0]*np.asarray(del_gripper_orientation)
    torque_control += np.asarray(inertial_1)*torque_control_gain[1]*(current_rotational_velocity-desired_rotational_velocity)
    
    return force_control, torque_control


def getCenterOfMassPosition(object_id):
    """Returns center of mass of the robot
    
        Returns:
            pos -- (x, y, z) robot center of mass
        """

    k = -1
    mass = 0
    com = np.array([0., 0., 0.])
    while True:
        if k == -1:
            pos, _ = pb.getBasePositionAndOrientation(object_id)
        else:
            res = pb.getLinkState(object_id, k)
            if res is None:
                break
            pos = res[0]

        d = pb.getDynamicsInfo(object_id, k)
        m = d[0]
        com += np.array(pos) * m
        mass += m

        k += 1
        
    return com / mass 

if (__name__ == '__main__'):

    # Parse arguments
    parser = argparse.ArgumentParser(description='Control mode selector')
    parser.add_argument('--control_mode', type=str, choices=['torque', 'angle'], required=True)
    args = parser.parse_args()
    control_mode = args.control_mode
    
    ## Initialize pybullet environment
    pb_client = bullet_client.BulletClient(connection_mode=pb.GUI)
    pb_client.setGravity(0, 0, -10)

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

    pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    ## Load URDF models
    plane_id = pb_client.loadURDF("plane.urdf")
    robot_id = pb_client.loadURDF('robot_models_new/kuka_gripper.urdf',
                                      basePosition=(0, 0, 0),
                                      globalScaling=1.0,
                                      useFixedBase=True)
    
    block_ids = []

    block_ids.append(pb_client.loadURDF('parts/zenga_block.urdf',
                                        basePosition=(.35, .4, .1),
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                                        # baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                                        globalScaling=1,
                                        useFixedBase=False))
    


    with open('gripper_polyfit_coefficient.npy', 'rb') as f:
        poly_coefficient = np.load(f)

    gripper_control_map = np.poly1d(poly_coefficient)
        
    gripper_angle = gripper_control_map(0.10482201257840675)
    joint_values = grasping_control(gripper_angle)

    for _ in range(240):
        for i, value in enumerate(joint_values):
            pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                            jointIndex=8+i,
                                            targetPosition=value,
                                            controlMode=pb.POSITION_CONTROL)

        pb_client.stepSimulation()

    ## Draw frames on the gripper and object
    draw_frame(pb_client, plane_id, -1, axis_length=.5),
    draw_frame(pb_client, plane_id, -1, xyz=(0,.5,0), axis_length=.5),
    draw_frame(pb_client, robot_id, 7, xyz=(0,0,0.27), axis_length=.2)

    # draw_frame(pb_client, robot_id, 9, xyz=(0,.025,0.065), axis_length=.2)
    # draw_frame(pb_client, robot_id, 12, xyz=(0,.025,0.065), axis_length=.2)
        
    left_finger_tip = pb.multiplyTransforms(positionA=pb_client.getLinkState(robot_id, 9)[0],
                                            orientationA=pb_client.getLinkState(robot_id, 9)[1],
                                            positionB=(.01, .025, .065),
                                            orientationB=(0,0,0,1))[0]

    right_finger_tip = pb.multiplyTransforms(positionA=pb_client.getLinkState(robot_id, 12)[0],
                                             orientationA=pb_client.getLinkState(robot_id, 12)[1],
                                             positionB=(-.01, .025, .065),
                                             orientationB=(0,0,0,1))[0]
    
    ## Draw frames at the finger tips
    # draw_frame(pb_client, plane_id, -1, xyz=left_finger_tip, axis_length=.2)
    # draw_frame(pb_client, plane_id, -1, xyz=right_finger_tip, axis_length=.2)

    draw_frame(pb_client, block_ids[0], -1, axis_length=.2)

    # pb_client.resetDebugVisualizerCamera(
    #     cameraDistance=.5,
    #     cameraYaw=90, cameraPitch=-30,
    #     cameraTargetPosition=(0,0,1.5))

    # pdb.set_trace()

    ## Main loop
    block_id = block_ids[0]
    block_position, block_orientation = pb_client.getBasePositionAndOrientation(bodyUniqueId=block_id)
    block_orientation = pb_client.getEulerFromQuaternion(block_orientation)

    print(block_position)
    
    pb_client.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=90, cameraPitch=-30,
        cameraTargetPosition=block_position)

    # pb_client.resetDebugVisualizerCamera(
    #         cameraDistance=.5,
    #         cameraYaw=90, cameraPitch=89,
    #         cameraTargetPosition=block_position)

    baseline_angle = 0.5585993153435626
    moving_direction = np.array((0-.35, .5-.4))
    starting_point = np.array([0.35555556, 0.37777778])
    starting_point -= 0.01*np.array([np.cos(baseline_angle), np.sin(baseline_angle)])
    starting_point -= .4*moving_direction

    print("Starting point: ", starting_point)
    time.sleep(1)


    for n in range(1000):
        joint_values = inverse_kinematics(pb_client, robot_id, 7, (starting_point[0], starting_point[1], .32), (0, np.pi, -np.pi/2+baseline_angle))[:8]
        
        position = get_end_effector_pose(pb_client, robot_id)

        for i, value in enumerate(joint_values):
            joint_angle_control(pb_client, robot_id, i, value)

        end_effector_position, end_effector_orientation = \
            get_end_effector_pose(pb_client, robot_id)

        end_effector_orientation = pb.getEulerFromQuaternion(end_effector_orientation)
        
        print("End effector position: ", end_effector_position, "End effector orientation: ", end_effector_orientation)

        time.sleep(1/240)
        pb_client.stepSimulation()

    ## Pushing block
    for n in range(1000):
        num_waypoints = 1000
        end_effector_position = get_end_effector_pose(pb_client, robot_id)[0]
        end_effector_target_position = np.array((.01369325, .46733263, .32))
        waypoints = np.linspace(end_effector_position, end_effector_target_position, num_waypoints)

        if n == 0:
            joint_angles_target, joint_velocities_target, joint_accelerations_target = compute_joint_profile(pb_client, robot_id, waypoints)
 
        joint_angles = np.array(get_joint_angles(pb_client, robot_id))
        joint_velocities = np.array(get_joint_velocities(pb_client, robot_id))
        K_1, K_2 = compute_control_gain('accurate')
        joint_torques = compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target[n], joint_velocities, joint_velocities_target[n], joint_accelerations_target[n])
        
        if control_mode == 'torque':
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

        elif control_mode == 'angle':            
            for i, value in enumerate(joint_angles_target[n]):    
                joint_angle_control(pb_client, robot_id, i, value)

        end_effector_position, end_effector_orientation = \
            get_end_effector_pose(pb_client, robot_id)

        end_effector_orientation = pb.getEulerFromQuaternion(end_effector_orientation)
        
        print("End effector position: ", end_effector_position, "End effector orientation: ", end_effector_orientation)
        print("Norm difference of joint angle: ", np.linalg.norm(joint_angles-joint_angles_target[n]))


        block_position, block_orientation = \
            pb_client.getBasePositionAndOrientation(bodyUniqueId=block_id)
        
        if (np.linalg.norm(np.asarray(block_position[:2])-np.array((.3, .5))) < .0005):
            print("Finished")
            break


        #pb_client.resetDebugVisualizerCamera(
        #    cameraDistance=.5,
        #    cameraYaw=90, cameraPitch=89,
        #    cameraTargetPosition=block_position)

        time.sleep(1/240)
        pb_client.stepSimulation()

    for n in range(600):
        linear_jacobian, angular_jacobian = get_jacobian(pb_client,
                                                    robot_id,
                                                    end_effector_link_id,
                                                    arm_joint_indices)

        J_f = np.matrix(linear_jacobian+angular_jacobian)
        J_f_peudo = J_f.T*np.linalg.inv(J_f*J_f.T)

        gripper_linear_velocity = np.array((moving_direction[0], moving_direction[1], 0.0))

        if (np.linalg.norm(gripper_linear_velocity) > .001):
            gripper_linear_velocity /= 10*np.linalg.norm(gripper_linear_velocity)
        else:
            gripper_linear_velocity = np.array((0,0,0))

        gripper_rotational_velocity = np.array((0,0,0))

        del_joint_angles = J_f_peudo*np.matrix(np.concatenate((gripper_linear_velocity,
                                                               gripper_rotational_velocity))).T
        for i, value in enumerate(del_joint_angles):
            joint_velocity_control(pb_client, robot_id, i, 0)
            
        time.sleep(1/240)
        pb_client.stepSimulation()

