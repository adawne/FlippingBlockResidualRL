import numpy as np
from scipy.optimize import minimize

g = 9.81

# Objective function (minimizing sqrt of total kinetic energy + potential energy)
def objective(x, I, m):
    v_x0, v_y0, theta_0, omega, h_0 = x
    T_ascent = v_y0 / g
    T_descent = np.sqrt(2 * (h_0 + (v_y0**2 / (2 * g))) / g)
    T = T_ascent + T_descent
    v_f = np.sqrt(v_x0**2 + v_y0**2)
    rotational_ke = 0.5 * I * omega**2
    translational_ke = 0.5 * m * v_f**2
    potential_energy = m * g * h_0
    total_energy = np.sqrt(rotational_ke + translational_ke + potential_energy)
    return total_energy

# Constraint to ensure the block lands at pi (upright)
def constraint_theta_final(x):
    v_x0, v_y0, theta_0, omega, h_0 = x
    T_ascent = v_y0 / g
    T_descent = np.sqrt(2 * (h_0 + (v_y0**2 / (2 * g))) / g)
    T = T_ascent + T_descent
    return np.pi - (theta_0 + omega * T)

# Constraint to limit the angular velocity
def constraint_omega_limit(x):
    omega = x[3]
    return 4.45 - omega

# New constraints for v_x0 and v_y0 based on realistic joint velocities
def constraint_vx_realistic(x, q, J):
    v_x0 = x[0]
    max_joint_velocities = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])  # example joint velocity limits
    v_max = np.linalg.norm(np.dot(J, max_joint_velocities))
    return v_max - v_x0

def constraint_vy_realistic(x, q, J):
    v_y0 = x[1]
    max_joint_velocities = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])  # example joint velocity limits
    v_max = np.linalg.norm(np.dot(J, max_joint_velocities))
    return v_max - v_y0

# Bounds for the optimization variables
bnds = [(0.1, None), (0, 2), (0, 2 * np.pi / 3), (0, 4.45), (0.35, None)]

# Constraints for the optimization
constr = [
    {'type': 'eq', 'fun': constraint_theta_final},
    {'type': 'ineq', 'fun': constraint_omega_limit},
    # Realistic v_x0 and v_y0 constraints
    {'type': 'ineq', 'fun': lambda x: constraint_vx_realistic(x, q_initial, J_initial)},
    {'type': 'ineq', 'fun': lambda x: constraint_vy_realistic(x, q_initial, J_initial)}
]

# Example initial conditions and Jacobian (replace with your actual values)
q_initial = np.zeros(6)  # Initial joint configuration (example)
J_initial = np.eye(2, 6)  # Placeholder Jacobian (replace with your computed Jacobian)

# Optimization loop (as in your original code)
for _ in range(4):
    I, m, a, b = random_inertia()
    for x0 in x0_list:
        result = minimize(
            objective,
            x0,
            args=(I, m),
            method='SLSQP',
            bounds=bnds,
            constraints=constr
        )
        # Display results...
