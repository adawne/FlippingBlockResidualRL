import numpy as np
import mujoco
import os
import csv

from scipy.spatial.transform import Rotation as R

def quat_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    return 1.0 - dot_product

def quat_distance_new(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    # Ensure the dot product is within valid bounds to prevent numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = 2 * np.arccos(dot_product)
    return angle  # Angle in radians


class ActuatorController:
    def __init__(self, actuator_ids) -> None:
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])
        self.actuator_ids = actuator_ids

    def switch_to_position_controller(self, model, kp=5000, kv=500):
        self.gain[0] = kp
        self.bias[1] = -kp
        self.bias[2] = -kv
        self.update_actuator(model)

    def switch_to_velocity_controller(self, model, kv=500):
        self.gain[0] = kv
        self.bias[1] = 0
        self.bias[2] = -kv
        self.update_actuator(model)

    def switch_to_torque_controller(self, model):
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])
        self.update_actuator(model)

    def update_actuator(self, model):
        for actuator_id in self.actuator_ids:
            model.actuator(actuator_id).dynprm[:3] = self.dyn
            model.actuator(actuator_id).gainprm[:3] = self.gain
            model.actuator(actuator_id).biasprm[:3] = self.bias

    def print_actuator_parameters(self, model):
        for actuator_id in self.actuator_ids:
            dyn_params = model.actuator(actuator_id).dynprm[:3]
            gain_params = model.actuator(actuator_id).gainprm[:3]
            bias_params = model.actuator(actuator_id).biasprm[:3]

            print(f"Actuator ID: {actuator_id}")
            print(f"  Dyn Parameters: {dyn_params}")
            print(f"  Gain Parameters: {gain_params}")
            print(f"  Bias Parameters: {bias_params}")
            print("-----------------------------------")

def move_ee_to_point(model, data, target_position, target_orientation):
    joint_angles_target = diffik(model, data, target_position, target_orientation)
    data.ctrl[:6] = joint_angles_target[:6]

def set_joint_states(data, actuator_ids, joint_angles):
    for i, actuator_id in enumerate(actuator_ids):
        data.ctrl[actuator_id] = joint_angles[i]

def execute_mpc_trajectory(trajectory_file, data, index):
    qpos_log = []

    with open(trajectory_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qpos_values = [float(x) for x in row['QPos'].split(',')]  
            qpos_log.append(qpos_values[:6])  # Only the first 6 qpos values

    # Convert qpos_log to a numpy array for easier indexing
    if index < len(qpos_log):
        qpos_log = np.array(qpos_log)
        mpc_qpos = qpos_log[index]
        data.qpos[:6] = mpc_qpos


def get_ee_pose(model, data):
    # TO DO: use pinch site
    end_effector_position = data.site('attachment_site').xpos
    end_effector_orientation_matrix = data.site('attachment_site').xmat.reshape(3, 3)
    
    end_effector_orientation = R.from_matrix(end_effector_orientation_matrix).as_euler('xyz')
    
    return end_effector_position, end_effector_orientation

def get_ee_velocity(model, data, local_frame=False):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
    
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, local_frame)

    return velocity.reshape(2, 3)[::-1]  


def get_block_pose(model, data, block_name='block_0', quat=False):
    block_id = data.body(block_name).id
    block_position = data.body(block_id).xpos.copy()

    if not quat:
        block_orientation_matrix = data.body(block_id).xmat.reshape(3, 3)
        block_orientation = R.from_matrix(block_orientation_matrix).as_euler('xyz')
    else:
        block_orientation = data.body(block_id).xquat.copy()

    return block_position, block_orientation


def get_block_velocity(data):
    block_translational_velocity = data.qvel[14:17]
    block_rotational_velocity = data.qvel[17:20]

    return block_translational_velocity, block_rotational_velocity
    
def get_joint_angles(data):
    return data.qpos[:6]

def get_specific_joint_angles(data, joint_ids):
    return [data.qpos[joint_id] for joint_id in joint_ids]

def get_joint_velocities(data):
    return data.qvel[:6]

def get_specific_joint_velocities(data, joint_ids):
    return [data.qvel[joint_id] for joint_id in joint_ids]

def hold_position(model, data, joint_angles, wait_time=1000):
    data.ctrl[:6] = joint_angles
    wait(model, data, wait_time)

def wait(model, data, wait_time):
    for _ in range(wait_time):
        mujoco.mj_step(model, data)

def gripper_open(data):
    data.ctrl[6] = 0

def gripper_close(data):
    data.ctrl[6] = 225

def diffik(model, data, target_position, target_orientation_euler, site="attachment_site"):
    target_orientation = R.from_euler('xyz', target_orientation_euler).as_quat()

    # Integration timestep in seconds. This corresponds to the amount of time the joint
    # velocities will be integrated for to obtain the desired joint positions.
    integration_dt: float = 1.0

    # Damping term for the pseudoinverse. This is used to prevent joint velocities from
    # becoming too large when the Jacobian is close to singular.
    damping: float = 1e-4

    # Whether to enable gravity compensation.
    gravity_compensation: bool = True

    # Simulation timestep in seconds.
    #dt: float = 0.002
    dt: float = model.opt.timestep

    # Maximum allowable joint velocity in rad/s. Set to 0 to disable.
    max_angvel = 0.0

    # Override the simulation timestep.
    #model.opt.timestep = dt

    # Name of bodies we wish to apply gravity compensation to.
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0



    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Position error.
    site_id = model.site(site).id
    error_pos[:] = target_position - data.site(site_id).xpos
    #error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos


    # Orientation error.
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    #mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_mulQuat(error_quat, target_orientation, site_quat_conj)
    mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

    # Get the Jacobian with respect to the end-effector site.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # Solve system of equations: J @ dq = error.
    dq = (jac.T @ np.linalg.solve(jac @ jac.T + diag, error))
    

    # Scale down joint velocities if they exceed maximum.
    if max_angvel > 0:
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

    # Integrate joint velocities to obtain joint positions.
    q = data.qpos.copy()
    mujoco.mj_integratePos(model, q, dq, integration_dt)

    # Set the control signal.
    np.clip(q[:15], *model.jnt_range.T, out=q[:15])

    return q

def generate_trajectory(start, end):
    num_points = np.linalg.norm((end - start)) / 0.01
    num_points = int(num_points)
    points = np.linspace(start, end, num_points)
    
    return points

def has_block_flipped(initial_quat, current_quat, tolerance_angle=5):
    initial_rot = R.from_quat(initial_quat)
    current_rot = R.from_quat(current_quat)
    relative_rotation = current_rot * initial_rot.inv()
    rotation_angle = relative_rotation.magnitude()
    rotation_axis = relative_rotation.as_rotvec()
    rotation_angle_degrees = np.degrees(rotation_angle)
    if 180 - tolerance_angle <= rotation_angle_degrees <= 180 + tolerance_angle:
        normalized_axis = rotation_axis / np.linalg.norm(rotation_axis)
        if np.isclose(abs(normalized_axis[0]), 1, atol=0.1):
            return True
        elif np.isclose(abs(normalized_axis[1]), 1, atol=0.1):
            return True
        elif np.isclose(abs(normalized_axis[2]), 1, atol=0.1):
            return True
    return False


def has_rotated(orientation_history):
    initial_orientation = R.from_euler('zyx', orientation_history[0])
    total_rotation_angle = 0

    for i in range(1, len(orientation_history)):
        current_orientation = R.from_euler('zyx', orientation_history[i])
        
        relative_rotation = initial_orientation.inv() * current_orientation
        
        rotation_angle = relative_rotation.magnitude() * (180 / np.pi)
        total_rotation_angle += rotation_angle
        initial_orientation = current_orientation

    return total_rotation_angle >= 360

def has_block_landed(data):
    floor_id = data.geom("floor").id
    block_id = data.geom("blue_subbox").id

    # Loop through all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        if ((geom1_id == floor_id and geom2_id == block_id) or (geom1_id == block_id and geom2_id == floor_id)):
            return True
    
    return False


def detect_block_contact(data):
    """
    Detects contact between the block and the robot, ignoring all contacts involving the floor.
    Returns a list of lists with geom1_id and geom2_id for each contact.
    """
    floor_id = data.geom("floor").id

    contacts = []

    for i in range(data.ncon):
        contact = data.contact[i]
        
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Ignore all contacts involving the floor
        if geom1_id != floor_id and geom2_id != floor_id:
            contacts.append([geom1_id, geom2_id])

    return contacts

def has_block_released(data):
    blue_subbox_id = data.geom("blue_subbox").id
    right_pad1_id = data.geom("right_pad1").id
    right_pad2_id = data.geom("right_pad2").id
    left_pad1_id = data.geom("left_pad1").id
    left_pad2_id = data.geom("left_pad2").id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2

        if (geom1_id in {blue_subbox_id, right_pad1_id, right_pad2_id, left_pad1_id, left_pad2_id} or
            geom2_id in {blue_subbox_id, right_pad1_id, right_pad2_id, left_pad1_id, left_pad2_id}):
            return False  

    return True  

def has_block_hit_floor(data):
    """
    Checks if the block has made contact with the floor.
    Returns True if a collision is detected between the block and the floor, False otherwise.
    """
    block_id = data.geom("blue_subbox").id  # Assuming "blue_subbox" represents the block
    floor_id = data.geom("floor").id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2

        # Check if the block is in contact with the floor
        if (geom1_id == block_id and geom2_id == floor_id) or \
           (geom2_id == block_id and geom1_id == floor_id):
            return True

    return False


def has_block_hit_robot(data):
    blue_subbox_id = data.geom("blue_subbox").id
    floor_id = data.geom("floor").id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2

        if (geom1_id not in {blue_subbox_id, floor_id} and
            geom2_id == blue_subbox_id) or \
           (geom2_id not in {blue_subbox_id, floor_id} and
            geom1_id == blue_subbox_id):
            return True

    return False         

def is_block_declining(block_height_hist):
    for i in range(1, len(block_height_hist)):
        if block_height_hist[i] < block_height_hist[i-1]:
            return True
        else:
            return False
    
def reset_block_position(model, data):
    target_position = np.array([0.5, 0.5, 0.1])
    target_orientation = np.array([np.pi/2, 0, 0])
    target_orientation_quat = R.from_euler('xyz', target_orientation).as_quat()

    data.joint("block_joint").qpos = np.concatenate((target_position, target_orientation_quat))
    data.joint("block_joint").qvel = np.zeros(6)
    #data.qpos[14:17] = target_position
    #data.qpos[17:21] = target_orientation_quat

def map_to_velocity(action):
    if action == 0:
        return 0
    else:
        return -action*np.pi/20