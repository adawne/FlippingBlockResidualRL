import mujoco
import numpy as np

def calculate_joint_velocities_full(model, data, desired_velocity, desired_orientation):
    site_id = model.site("attachment_site").id
    
    jac = np.zeros((6, model.nv))
    
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    desired_vel_orientation = np.concatenate((desired_velocity, desired_orientation))
    
    damping = 1e-4
    diag_damping = damping * np.eye(6)
    
    joint_velocities = jac.T @ np.linalg.solve(jac @ jac.T + diag_damping, desired_vel_orientation)
    
    joint_velocities = joint_velocities[:6]

    joint_vel_limits = np.array([2*np.pi/3, 2*np.pi/3, np.pi, np.pi, np.pi, np.pi]) 
    joint_velocities = np.clip(joint_velocities, -joint_vel_limits, joint_vel_limits)
    
    return joint_velocities


def calculate_joint_velocities_trans_partial(model, data, desired_velocity, desired_orientation):
    site_id = model.site("attachment_site").id
    
    jac_trans = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_trans, None, site_id)
    
    joint_indices = [1, 2]
    jac_trans_reduced = jac_trans[:, joint_indices]
    
    damping = 1e-4
    diag_damping = damping * np.eye(3)
    
    joint_velocities_reduced = jac_trans_reduced.T @ np.linalg.solve(jac_trans_reduced @ jac_trans_reduced.T + diag_damping, desired_velocity)
    
    joint_velocities = np.zeros(model.nv)
    joint_velocities[joint_indices] = joint_velocities_reduced
    
    joint_vel_limits = np.array([2*np.pi/3, np.pi/2]) 
    joint_velocities[joint_indices] = np.clip(joint_velocities[joint_indices], -joint_vel_limits, joint_vel_limits)
    
    return joint_velocities[:6]



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
