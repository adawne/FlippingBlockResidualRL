import mujoco
import numpy as np

def calculate_joint_velocities_test(model, data, desired_velocity, desired_orientation):
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

def calculate_joint_velocities(model, data, desired_velocity, desired_orientation):
    site_id = model.site("attachment_site").id
    
    # Jacobian for translational velocity (3x6 matrix for 6 DoF)
    jac_trans = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_trans, None, site_id)
    
    damping = 1e-4
    diag_damping = damping * np.eye(3)
    
    # Calculate joint velocities considering only translational velocity
    joint_velocities = jac_trans.T @ np.linalg.solve(jac_trans @ jac_trans.T + diag_damping, desired_velocity)
    
    # Consider only the first 6 joint velocities (assuming a 6-DoF robot)
    joint_velocities = joint_velocities[:6]
    
    # Joint velocity limits
    joint_vel_limits = np.array([2*np.pi/3, 2*np.pi/3, np.pi, np.pi, np.pi, np.pi]) 
    joint_velocities = np.clip(joint_velocities, -joint_vel_limits, joint_vel_limits)
    
    return joint_velocities
