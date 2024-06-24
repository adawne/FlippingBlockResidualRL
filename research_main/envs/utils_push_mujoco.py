import numpy as np
import mujoco
import os
import ikpy.chain
import roboticstoolbox as rtb
import ur10e_ikfast

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from ur_ikfast import ur_kinematics

def create_ur_model(marker_position=None, block_positions_orientations=None):
    marker_body = ""
    if marker_position is not None:
        marker_body = f'''
        <body name="marker" pos="{marker_position[0]} {marker_position[1]} {marker_position[2]}">
            <geom size="0.01 0.01 0.01" type="sphere" rgba="1 0 0 1"/>
        </body>'''

    block_bodies = ""
    if block_positions_orientations is not None:
        for i, (pos, euler) in enumerate(block_positions_orientations):
            # Convert Euler angles to quaternion
            quat = R.from_euler('zyx', euler).as_quat()
            block_bodies += f'''
            <body name="block{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}">
                <joint type="free" name="free_joint_{i}" damping="1.85"/>
                <geom size="0.05 0.02 0.08" type="box" rgba="0.8 0.5 0.3 1" friction="2 0.001 0.0001"/>
            </body>'''

    xml_string = f'''
    <mujoco model="ur10e scene">
        <include file="universal_robots_ur10e_2f85/ur10e_2f85.xml"/>

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
            {marker_body}
            {block_bodies}
        </worldbody>
    </mujoco>'''
    
    return xml_string


class ActuatorController:
    def __init__(self, actuator_ids) -> None:
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])
        self.actuator_ids = actuator_ids

    def set_position_controller(self, kp=5000, kv=500):
        self.gain[0] = kp
        self.bias[1] = -kp
        self.bias[2] = -kv

    def set_velocity_controller(self, kv=500):
        self.gain[0] = kv
        self.bias[2] = -kv

    def set_torque_controller(self):
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])

    def update_actuator(self, model):
        for actuator_id in self.actuator_ids:
            model.actuator(actuator_id).dynprm[:3] = self.dyn
            model.actuator(actuator_id).gainprm[:3] = self.gain
            model.actuator(actuator_id).biasprm[:3] = self.bias


def get_ee_pose(model, data):
    end_effector_id = model.body('wrist_3_link').id
    end_effector_position = data.site('attachment_site').xpos
    end_effector_orientation = data.body(end_effector_id).xquat
    end_effector_orientation = R.from_quat(end_effector_orientation).as_euler('zyx')
    
    return end_effector_position, end_effector_orientation

def get_block_pose(model, data, block_name):
    block_id = model.body(block_name).id
    block_position = data.body(block_id).xpos
    block_orientation = data.body(block_id).xquat
    block_orientation = R.from_quat(block_orientation).as_euler('zyx')
    
    return block_position, block_orientation

def get_joint_angles(data):
    return data.qpos[:6]

def hold_position(model, data, joint_angles, wait_time=1000):
    data.ctrl[:6] = joint_angles
    wait(model, data, wait_time)

def wait(model, data, wait_time):
    for _ in range(wait_time):
        mujoco.mj_step(model, data)

def gripper_open(data):
    data.ctrl[6] = 25

def gripper_close(data):
    data.ctrl[6] = 200

def diffik(model, data, target_position, target_orientation_euler):
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
    dt: float = 0.002

    # Maximum allowable joint velocity in rad/s. Set to 0 to disable.
    max_angvel = 0.0

    # Override the simulation timestep.
    model.opt.timestep = dt

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
    site_id = model.site("attachment_site").id
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