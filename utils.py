import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import cvxpy as cp
import pickle

from kuka_dynamics import *
from kuka_kinematics import *
from torque_controller import *
import control

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
end_effector_link_id = 10
constraint_id = -1
grasped_block_id = -1
block_ids = []
new_block_location = (.5, 0, .1)

def set_camera(distance, yaw, pitch, position):
    pb.resetDebugVisualizerCamera(cameraDistance=distance,
                                  cameraYaw=yaw,
                                  cameraPitch=pitch,
                                  cameraTargetPosition=position)

def initializeGUI(enable_gui=True, gravity=-10):
    pb_client = bc.BulletClient(connection_mode=pb.GUI)
    
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, enable_gui)

    pb_client.resetDebugVisualizerCamera(cameraDistance=2,
                                         cameraYaw=0,
                                         cameraPitch=-20,
                                         cameraTargetPosition=(0,0,.5))

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    pb_client.setGravity(0,0,gravity)

    return pb_client

def draw_frame(pb_client, robot_id, link_index, axis_length=.2, line_width=1, line_from=(0,0,0)):
    pb_client.addUserDebugLine(lineFromXYZ=line_from,
                               lineToXYZ=line_from+np.array((axis_length,0,0)),
                               lineColorRGB=(1,0,0),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=line_from,
                               lineToXYZ=line_from+np.array((0,axis_length,0)),
                               lineColorRGB=(0,1,0),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=line_from,
                               lineToXYZ=line_from+np.array((0,0,axis_length)),
                               lineColorRGB=(0,0,1),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)
    
def mark_point(pb_client, point, axis_length=.2, line_width=1):
    point = np.array(point)
    
    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(axis_length,0,0),
                               lineWidth=line_width,
                               lineColorRGB=(1,0,0))

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,axis_length,0),
                               lineWidth=line_width,
                               lineColorRGB=(0,1,0))

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,0,axis_length),
                               lineWidth=line_width,
                               lineColorRGB=(0,0,1))
    
def add_debug_parameters(pb_client, parameter_info):
    debug_parameter_ids = []
    for data in parameter_info:
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=data['name'],
                                                             rangeMin=data['lower_limit'],
                                                             rangeMax=data['upper_limit'],
                                                             startValue=data['start_value'])
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

def add_joint_debug_parameters(pb_client, arm_joint_indices, gripper_joint_indices=[]):
    debug_parameter_ids = []

    for joint_index in arm_joint_indices:
        joint_name = 'arm joint {}'.format(joint_index+1)
        joint_lower_limit = -180
        joint_upper_limit = 180
        start_value = (joint_lower_limit+joint_upper_limit)/2
        
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=joint_name,
                                                             rangeMin=joint_lower_limit,
                                                             rangeMax=joint_upper_limit,
                                                             startValue=start_value)
        
        debug_parameter_ids.append(debug_parameter_id)

    for joint_index in gripper_joint_indices:
        joint_name = 'gripper joint {}'.format(joint_index+1)
        joint_lower_limit = -180
        joint_upper_limit = 180
        start_value = (joint_lower_limit+joint_upper_limit)/2
        
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=joint_name,
                                                             rangeMin=joint_lower_limit,
                                                             rangeMax=joint_upper_limit,
                                                             startValue=start_value)
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids


def get_jacobian(pb_client, robot_id, link_id, joint_angles):
    #joint_angles = get_joint_angles(pb_client, robot_id)
    joint_angles = list(joint_angles) + [0]*2
    
    linear_jacobian, angular_jacobian = \
        pb_client.calculateJacobian(bodyUniqueId=robot_id,
                                    linkIndex=link_id,
                                    localPosition=(0,0,0),
                                    objPositions=joint_angles,
                                    objVelocities=[0]*len(joint_angles),
                                    objAccelerations=[0]*len(joint_angles))

    return linear_jacobian, angular_jacobian
        
def get_end_effector_pose(pb_client, robot_id):
    link_world_position, link_world_orientation, _, _, _, _ = \
        pb_client.getLinkState(bodyUniqueId=robot_id,
                               linkIndex=end_effector_link_id)

    return np.array(link_world_position), np.array(link_world_orientation)

def place_new_block(pb_client):
    global block_ids

    for block_id in block_ids:
        position, orientation = get_block_pose(pb_client, block_id)

        if (np.linalg.norm(np.asarray(position)-new_block_location) < .5):
            return -1
    
    block_id = pb_client.loadURDF('./robot_models/zenga_block.urdf',
                                  basePosition=new_block_location,
                                  baseOrientation=\
                                  pb.getQuaternionFromEuler((np.pi, 0, np.pi)))
    pb_client.changeDynamics(block_id,
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

    block_ids.append(block_id)

    return block_id

def get_block_pose(pb_client, block_id):

    position, orientation = pb_client.getBasePositionAndOrientation(block_id)

    return position, orientation

def gripper_open(pb_client, robot_id):
    gripper_value = [0.1,0.1]
    for i, value in enumerate(gripper_value):
        pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                        jointIndex=i+8,
                                        targetPosition=value,
                                        controlMode=pb.POSITION_CONTROL,
                                        force=500)

def gripper_close(pb_client, robot_id):
    gripper_value = [0, 0]
    for i, value in enumerate(gripper_value):
        pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                        jointIndex=i+8,
                                        targetPosition=value,
                                        controlMode=pb.POSITION_CONTROL,
                                        force=500)

def grasp_object(pb_client, robot_id, K_1, K_2, joint_angles, joint_angles_target, joint_velocities):
    global constraint_id, grasped_block_id
    
    if (constraint_id >= 0):
        return grasped_block_id

    end_effector_position, end_effector_orientation = get_end_effector_pose(pb_client, robot_id)

    for block_id in block_ids:
        block_position, block_orientation = get_block_pose(pb_client, block_id)

        if (np.linalg.norm(end_effector_position-block_position) < .05 and
            np.linalg.norm(end_effector_orientation-block_orientation) < .05):
            
            constraint_id = pb_client.createConstraint(parentBodyUniqueId=robot_id,
                                                       parentLinkIndex=end_effector_link_id,
                                                       childBodyUniqueId=block_id,
                                                       childLinkIndex=-1,
                                                       jointType=pb.JOINT_FIXED,
                                                       jointAxis=[0,0,0],
                                                       parentFramePosition=[0, 0, 0],
                                                       childFramePosition=[0, 0 ,0],
                                                       childFrameOrientation=pb.getQuaternionFromEuler((0, 0, 0)))

            
            for _ in range(50):
                gripper_close(pb_client, robot_id)
                joint_torques = compute_joint_torques_move_to_point(pb_client, robot_id, K_1, K_2, joint_angles, joint_angles_target, joint_velocities)
                for i, joint_index in enumerate(arm_joint_indices):
                    joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])
                pb_client.stepSimulation()
            
            grasped_block_id = block_id
            return grasped_block_id

    return -1        


def grasp_block(pb_client, robot_id):
    global constraint_id, grasped_block_id
    
    if (constraint_id >= 0):
        return grasped_block_id

    end_effector_position, end_effector_orientation = get_end_effector_pose(pb_client, robot_id)

    for block_id in block_ids:
        block_position, block_orientation = get_block_pose(pb_client, block_id)

        if (np.linalg.norm(end_effector_position-block_position) < .05 and
            np.linalg.norm(end_effector_orientation-block_orientation) < .05):
            
            constraint_id = pb_client.createConstraint(parentBodyUniqueId=robot_id,
                                                       parentLinkIndex=end_effector_link_id,
                                                       childBodyUniqueId=block_id,
                                                       childLinkIndex=-1,
                                                       jointType=pb.JOINT_FIXED,
                                                       jointAxis=[0,0,0],
                                                       parentFramePosition=[0, 0, 0],
                                                       childFramePosition=[0, 0 ,0],
                                                       childFrameOrientation=pb.getQuaternionFromEuler((0, 0, 0)))

            
            # for _ in range(50):
            #     gripper_close(pb_client, robot_id)
            #     pb_client.stepSimulation()
            
            grasped_block_id = block_id
            return grasped_block_id

    return -1        


def release_object(pb_client, robot_id, K_1, K_2, joint_angles, joint_angles_target, joint_velocities):
    global constraint_id, grasped_block_id
    if (constraint_id >= 0):
        for _ in range(50):
            gripper_open(pb_client, robot_id)
            joint_torques = compute_joint_torques_move_to_point(pb_client, robot_id, K_1, K_2, joint_angles, joint_angles_target, joint_velocities)
            for i, joint_index in enumerate(arm_joint_indices):
                joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])
            pb_client.stepSimulation()

        joint_torques = compute_joint_torques_move_to_point(pb_client, robot_id, K_1, K_2, joint_angles, joint_angles_target, joint_velocities)
        for i, joint_index in enumerate(arm_joint_indices):
            joint_torque_control(pb_client, robot_id, joint_index, joint_torques[i])

                
        pb_client.removeConstraint(constraint_id)
        # pb_client.stepSimulation()
        
        constraint_id = -1
        grasped_block_id = -1

def release_block(pb_client, robot_id):
    global constraint_id, grasped_block_id

    if (constraint_id >= 0):
        # for _ in range(50):
        #     gripper_open(pb_client, robot_id)
        #     pb_client.stepSimulation()
                
        pb_client.removeConstraint(constraint_id)
        # pb_client.stepSimulation()
        
        constraint_id = -1
        grasped_block_id = -1



def get_joint_info(pb_client, robot_id):
    number_of_joints = pb_client.getNumJoints(bodyUniqueId=robot_id)
    joint_info = []
    for joint_index in range(number_of_joints):
        return_data = pb_client.getJointInfo(bodyUniqueId=robot_id,
                                             jointIndex=joint_index)
    
        joint_index, joint_name = return_data[:2]
        joint_lower_limit = return_data[8]
        joint_upper_limit = return_data[9]
        joint_info.append({'index': joint_index,
                           'name': joint_name,
                           'limit': (joint_lower_limit, joint_upper_limit)})

    return joint_info

def get_joint_angles(pb_client, robot_id):
    joint_angles = []
    for joint_index in arm_joint_indices:
        position, velocity, force, torque = pb_client.getJointState(bodyUniqueId=robot_id,
                                                                    jointIndex=joint_index)

        joint_angles.append(position)

    return joint_angles

def get_joint_velocities(pb_client, robot_id):
    joint_velocities = []
    for joint_index in arm_joint_indices:
        position, velocity, force, torque = pb_client.getJointState(bodyUniqueId=robot_id,
                                                                    jointIndex=joint_index)

        joint_velocities.append(velocity)

    return joint_velocities

def free_joint_torques(pb_client, robot_id, joint_indices):
    pb_client.setJointMotorControlArray(bodyUniqueId=robot_id,
                                        jointIndices=joint_indices,
                                        controlMode=pb.VELOCITY_CONTROL,
                                        forces=np.zeros_like(joint_indices))

    
def joint_angle_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    targetPosition=value,
                                    controlMode=pb.POSITION_CONTROL,
                                    maxVelocity=1,
                                    force=500)

    return

def joint_velocity_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    targetVelocity=value,
                                    controlMode=pb.VELOCITY_CONTROL)

    return

def joint_torque_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    force=value,
                                    controlMode=pb.TORQUE_CONTROL)
    
def get_contact_points(pb_client, robot_id, obstacle_id):
    return pb_client.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)

def collision_check(pb_client, robot_id, obstacles_id):
    for obstacle_id in obstacles_id:
        contact_points = get_contact_points(pb_client, robot_id, obstacle_id)

        if (len(contact_points) > 0):
            return True

    return False

def check_object_graspability(pb_client, robot_id, object_id):
    linear_velocity, angular_velocity = pb_client.getLinkState(bodyUniqueId=robot_id,
                                                               linkIndex=end_effector_link_id,
                                                               computeLinkVelocity=1)[6:]

    if (np.linalg.norm(linear_velocity) > 2):
        return False

    return True


def update_block_target(floor, side):
    length = 0.18
    width = 0.06
    height = 0.03
    
    if floor % 2 == 0:  # Even floors
        print("even")
        block_target_position = [-.2, -.25 + (side *2* width), floor * height]
        block_target_orientation = (np.pi, 0, 0)
    else:  # Odd floors
        print("odd")
        if side == 0:
            side = -1
        block_target_position = [-.2 + (side * width), -.25 + width, floor * height]
        block_target_orientation = (0, 0, 0)

    return block_target_position, block_target_orientation



def inverse_kinematics(pb_client, robot_id, link_id, target_position, target_orientation):
    target_quaternion = pb.getQuaternionFromEuler(target_orientation)
    joint_values = pb_client.calculateInverseKinematics(bodyUniqueId=robot_id,
                                                        endEffectorLinkIndex=link_id,
                                                        targetPosition=target_position,
                                                        targetOrientation=target_quaternion)

    return joint_values

def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data_with_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def velocity_expectation(pb_client, robot_id, end_effector_link_id, joint_angles_target, joint_velocities_target):
    expected_velocity = []
    for i in range (len(joint_angles_target)-1):
        theta_i = joint_angles_target[i]
        theta_dot_i = joint_velocities_target[i]
        J = compute_jacobian(pb_client, robot_id, end_effector_link_id, theta_i)
        v = np.linalg.norm(J @ theta_dot_i)
        expected_velocity.append(v)

    expected_velocity.append(0)

    return expected_velocity
    
def velocity_checker(pb_client, robot_id, end_effector_link_id, state_iteration, expected_velocity, highest_velocity, lowest_velocity):

    linear_velocity, _ = pb_client.getLinkState(bodyUniqueId=robot_id, linkIndex=end_effector_link_id, computeLinkVelocity=1)[6:]
    linear_velocity_ee = np.linalg.norm(linear_velocity)

    if linear_velocity_ee > highest_velocity:
        highest_velocity = linear_velocity_ee

    if linear_velocity_ee < lowest_velocity and linear_velocity_ee > 0.1:
        lowest_velocity = linear_velocity_ee

    #print("End effector velocity: ", linear_velocity_ee)
    #print("Expected velocity: ", expected_velocity[state_iteration])
    #print("Difference between expected and actual end effector velocities: ", np.linalg.norm(linear_velocity_ee - expected_velocity[state_iteration]))

    return lowest_velocity, highest_velocity

def compute_jacobian(pb_client, robot_id, end_effector_link_id, joint_angles):
    linear_jacobian, _ = get_jacobian(pb_client, robot_id, end_effector_link_id, joint_angles)
    J = np.array(linear_jacobian)[:, :7]
    return J






