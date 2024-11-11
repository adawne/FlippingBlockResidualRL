import numpy as np
from scipy.optimize import minimize

g = 9.81

# Objective function (minimizing sqrt of total kinetic energy + potential energy)
def objective(x, I, m):
    v_x0, v_y0, theta_0, omega, h_0 = x
    # Time of flight based on correct parabolic motion
    T_ascent = v_y0 / g
    T_descent = np.sqrt(2 * (h_0 + (v_y0**2 / (2 * g))) / g)
    T = T_ascent + T_descent
    
    # Total translational velocity at release
    v_f = np.sqrt(v_x0**2 + v_y0**2) 
    
    # Rotational kinetic energy
    rotational_ke = 0.5 * I * omega**2
    
    # Translational kinetic energy
    translational_ke = 0.5 * m * v_f**2
    
    # Potential energy due to release height
    potential_energy = m * g * h_0
    
    # Total energy (objective to minimize)
    total_energy = np.sqrt(rotational_ke + translational_ke + potential_energy)
    
    return total_energy

# Add regularization to objective to ensure smooth optimization
def objective_with_regularization(x, I, m):
    regularization = 1e-6 * np.sum(x**2)
    return objective(x, I, m) + regularization

# Constraint to ensure the block lands at pi (upright)
def constraint_theta_final(x):
    v_x0, v_y0, theta_0, omega, h_0 = x
    T_ascent = v_y0 / g
    
    descent_first_term = v_y0**2 / g**2
    descent_second_term = 2 * h_0 / g
    T_descent = np.sqrt(descent_first_term + descent_second_term)
    T = T_ascent + T_descent
    return np.pi - (theta_0 + omega * T)

# Constraint to limit the angular velocity
def constraint_omega_limit(x):
    omega = x[3]
    return 4.45 - omega  # omega <= 4.45 rad/s

# Bounds for the optimization variables
bnds = [
    (0.1, None),              # v_x0 >= 0.1
    (0, 2),              # v_y0 >= 0
    (0, 2 * np.pi / 3),     # 0 <= theta_0 <= 2*pi/3
    (0, 4.45),              # 0 <= omega <= 4.45
    (0.35, None)               # h_0 >= 0 (release height)
]

# Constraints for the optimization
constr = [
    {'type': 'eq', 'fun': constraint_theta_final},
    {'type': 'ineq', 'fun': constraint_omega_limit}
]

# Random inertia generation for each block
def random_inertia():
    a = np.random.uniform(0.05, 0.2)
    b = np.random.uniform(0.05, 0.2)
    m = np.random.uniform(0.5, 2.0)
    I = (1 / 12) * m * (a**2 + b**2)
    return I, m, a, b

# List of initial guesses for the optimization
x0_list = [
    [5, 5, np.pi / 6, 2, 1],  # initial guesses with release height h_0
    [3, 2, np.pi / 4, 1.5, 2],
    [8, 6, np.pi / 3, 3, 3],
    [6, 4, np.pi / 5, 2.5, 1.5]
]

# Loop to perform the optimization for random blocks
for _ in range(4):
    I, m, a, b = random_inertia()
    print(f"\nRandomized dimensions and mass: a = {a:.3f}m, b = {b:.3f}m, mass = {m:.3f}kg, inertia = {I:.5f} kg·m²")
    
    for x0 in x0_list:
        result = minimize(
            objective_with_regularization,
            x0,
            args=(I, m),
            method='SLSQP',
            bounds=bnds,
            constraints=constr
        )
        v_x0_opt, v_y0_opt, theta_0_opt, omega_opt, h_0_opt = result.x
        gradient = result.jac
        
        # Calculate time of flight T based on optimized vertical velocity and height
        T_ascent = v_y0_opt / g
        T_descent = np.sqrt(2 * (h_0_opt + (v_y0_opt**2 / (2 * g))) / g)
        T = T_ascent + T_descent
        
        # Calculate the required omega to make sure the block lands upright
        omega_required = (np.pi - theta_0_opt) / T
        
        # Check if the optimized omega matches the required omega
        omega_match = np.isclose(omega_opt, omega_required, rtol=1e-2)
        
        print("===========================")
        print(f"Initial guess: {x0}")
        print(f"Optimized values: v_x0 = {v_x0_opt:.4f}, v_y0 = {v_y0_opt:.4f}, theta_0 = {theta_0_opt:.4f}, omega = {omega_opt:.4f}, h_0 = {h_0_opt:.4f}")
        print(f"Objective value: {result.fun:.4f}")
        print(f"Gradient at the solution: {gradient}")
        print(f"Time of flight (T): {T:.4f} s")
        print(f"Required omega: {omega_required:.4f} rad/s")
        print(f"Does optimized omega match required omega? {'Yes' if omega_match else 'No'}")
        print(f"Will the block stand at the end (theta_f = pi)? {'Yes' if omega_match else 'No'}")
