import numpy as np
from scipy.optimize import minimize
import mujoco
import os

def compute_jacobian(model, data, q):
    data.qpos[:6] = q
    mujoco.mj_step(model, data)
    #print(data.qpos[:6].copy())  

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    jac = np.vstack([jacp[:, :6], jacr[:, :6]])
    return jac

def forward_kinematics(model, data, q):
    data.qpos[:6] = q
    mujoco.mj_step(model, data)
    pos = data.sensor('pinch_pos').data.copy() 
    ori = data.sensor('pinch_quat').data.copy()  
    return np.concatenate((pos, ori))


current_dir = os.path.dirname(__file__)
path_to_model = os.path.join(current_dir, '..', 'research_main', 'envs', 'universal_robots_ur10e_2f85_example', 'ur10e_2f85.xml')
model = mujoco.MjModel.from_xml_path(path_to_model)    
data = mujoco.MjData(model)

n = 6
q_min = np.array([-2 * np.pi] * n)
q_max = np.array([2 * np.pi] * n)
q_dot_min = np.array([-120 * np.pi / 180] * 2 + [-180 * np.pi / 180] * 4)
q_dot_max = np.array([120 * np.pi / 180] * 2 + [180 * np.pi / 180] * 4)
alpha = 1.0
beta = 1.0

x_ee_desired = np.array([0.9, 0.2, 0.35, 
                        -1.25300627e-04, -3.16130824e-01, -3.86373210e-04, -9.48715520e-01])
v_ee_desired = np.array([0.2, 0, 0.5864, 0, -3.14, 0])
#initial_q = np.zeros(n) 
initial_q = [-1.5923697502906031,-2.2355866024308724,2.702949141598393,-2.067142791964366,-1.572676790518878,-4.223297354370639e-09]


def objective(vars):
    q = vars[:n]
    q_dot = vars[n:]
    f_q = forward_kinematics(model, data, q)
    J = compute_jacobian(model, data, q)
    error_pose = np.linalg.norm(f_q - x_ee_desired)**2
    error_velocity = np.linalg.norm(J @ q_dot - v_ee_desired)**2
    return alpha * error_velocity + beta * error_pose

# Define the constraints
def joint_limits(vars):
    q = vars[:n]
    q_dot = vars[n:]
    return np.hstack([
        q - q_min,
        q_max - q,
        q_dot - q_dot_min,
        q_dot_max - q_dot
    ])

# Initial guess
initial_vars = np.hstack([initial_q, np.zeros(n)])

# Bounds
bounds = [(q_min[i], q_max[i]) for i in range(n)] + [(q_dot_min[i], q_dot_max[i]) for i in range(n)]

# Optimization
result = minimize(
    objective,
    initial_vars,
    method='SLSQP',
    bounds=bounds,
    constraints={'type': 'ineq', 'fun': joint_limits}
)

if result.success:
    q_optimal = result.x[:n]
    q_dot_optimal = result.x[n:]
    print(f"Q optimal: {q_optimal}")
    print(f"Q dot optimal: {q_dot_optimal}")

    ee_position = forward_kinematics(model, data, q_optimal)
    print("Optimal End-Effector Position (position and orientation):", ee_position)

    J_opt = compute_jacobian(model, data, q_optimal)

    ee_velocity = J_opt @ q_dot_optimal
    print("Optimal End-Effector Velocity:", ee_velocity)
else:
    print("Optimization failed:", result.message)
