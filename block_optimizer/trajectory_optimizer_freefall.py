import numpy as np
from scipy.optimize import minimize

g = 9.81

def objective(x, I, m):
    v_x0, v_y0, theta_0, omega, h = x
    T = np.sqrt(2 * h / g)  
    v_f = np.sqrt(v_x0**2 + v_y0**2) 
    potential_energy = m * g * h  
    return np.sqrt(I * omega**2 + m * v_f**2 + 10 * potential_energy)

def objective_with_regularization(x, I, m):
    regularization = 1e-6 * np.sum(x**2)
    return objective(x, I, m) + regularization

def constraint_theta_final(x):
    v_x0, v_y0, theta_0, omega, h = x
    T = np.sqrt(2 * h / g)  
    return np.pi - (theta_0 + omega * T) 

def constraint_omega_limit(x):
    omega = x[3]
    return (1.5) - omega  

def constraint_height_limit(x):
    h = x[4]
    return h - 0.1  

# Bounds for the optimization variables
bnds = [(0, None), (0, None), (0, 2 * np.pi / 3), (0,  1.5), (0.1, 1)] 

# Constraints for the optimization
constr = [
    {'type': 'eq', 'fun': lambda x: constraint_theta_final(x)},
    {'type': 'ineq', 'fun': lambda x: constraint_omega_limit(x)},
    {'type': 'ineq', 'fun': lambda x: constraint_height_limit(x)}  
]

# Random inertia generation for each block
def random_inertia():
    a = np.random.uniform(0.05, 0.2)
    b = np.random.uniform(0.05, 0.2)
    m = np.random.uniform(0.5, 2.0)
    I = (1 / 12) * m * (a**2 + b**2)
    return I, m, a, b

# List of initial guesses for the optimization (including height h as the last variable)
x0_list = [
    [0, 0, 0, 0, 0.3],
    [0, 0, 0, 0, 0.3],
    [0, 0, 0, 0, 0.3],
    [0, 0, 0, 0, 0.3]
]

# Loop to perform the optimization for random blocks
for _ in range(4):
    I, m, a, b = random_inertia()
    print(f"\nRandomized dimensions and mass: a = {a:.3f}m, b = {b:.3f}m, mass = {m:.3f}kg, inertia = {I:.5f} kg·m²")
    
    for x0 in x0_list:
        result = minimize(objective_with_regularization, x0, args=(I, m), method='trust-constr', bounds=bnds, constraints=constr)
        v_x0_opt, v_y0_opt, theta_0_opt, omega_opt, h_opt = result.x
        gradient = result.jac
        
        # Calculate time of flight T based on optimized height h
        T = np.sqrt(2 * h_opt / g)
        
        # Calculate the required omega to make sure the block lands upright
        omega_required = (np.pi - theta_0_opt) / T
        
        # Check if the optimized omega matches the required omega
        omega_match = np.isclose(omega_opt, omega_required, rtol=1e-2)
        
        print("===========================")
        print(f"Initial guess: {x0}")
        print(f"Optimized values: v_x0 = {v_x0_opt:.4f}, v_y0 = {v_y0_opt:.4f}, theta_0 = {theta_0_opt:.4f}, omega = {omega_opt:.4f}, h = {h_opt:.4f} m")
        print(f"Objective value: {result.fun:.4f}")
        print(f"Gradient at the solution: {gradient}")
        print(f"Time of flight (T): {T:.4f} s")
        print(f"Required omega: {omega_required:.4f} rad/s")
        print(f"Does optimized omega match required omega? {'Yes' if omega_match else 'No'}")
        print(f"Will the block stand at the end (theta_f = pi)? {'Yes' if omega_match else 'No'}")
