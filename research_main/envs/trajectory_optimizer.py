import numpy as np
from scipy.optimize import minimize

g = 9.81

def objective(x, I, m):
    v_x0, v_y0, theta_0 = x
    T = 2 * v_y0 / g
    omega = (2 * np.pi - theta_0) / T
    v_f = np.sqrt(v_x0**2 + (v_y0 + g * T)**2)
    return np.sqrt(I * omega**2 + m * v_f**2)

def objective_with_regularization(x, I, m):
    regularization = 1e-6 * np.sum(x**2)  # Add a small regularization term
    return objective(x, I, m) + regularization

def constraint_omega(x):
    v_x0, v_y0, theta_0 = x
    T = 2 * v_y0 / g
    omega = (2 * np.pi - theta_0) / T
    return 10 - omega

bnds = [(0.1, None), (0.1, None), (0, 2*np.pi/3)]

constr = [
    {'type': 'ineq', 'fun': lambda x: constraint_omega(x)}
]

# Function to generate random dimensions and mass
def random_inertia():
    a = np.random.uniform(0.05, 0.2)  # Random width between 0.05m and 0.2m
    b = np.random.uniform(0.05, 0.2)  # Random height between 0.05m and 0.2m
    m = np.random.uniform(0.5, 2.0)  # Random mass between 0.5kg and 2kg
    I = (1/12) * m * (a**2 + b**2)  # Moment of inertia for a rectangular block
    return I, m, a, b

x0_list = [
    [5, 5, np.pi / 2]
]

# Iterate over randomized inertia instead of initial guesses
for _ in range(4):  # Run 4 iterations with different random inertia
    I, m, a, b = random_inertia()
    print(f"\nRandomized dimensions and mass: a = {a:.3f}m, b = {b:.3f}m, mass = {m:.3f}kg, inertia = {I:.5f} kg·m²")
    
    for x0 in x0_list:
        result = minimize(objective_with_regularization, x0, args=(I, m), method='trust-constr', bounds=bnds, constraints=constr)
        v_x0_opt, v_y0_opt, theta_0_opt = result.x
        gradient = result.jac
        print("===========================")
        print(f"Initial guess: {x0}")
        print(f"Optimized values: v_x0 = {v_x0_opt:.4f}, v_y0 = {v_y0_opt:.4f}, theta_0 = {theta_0_opt:.4f}")
        print(f"Objective value: {result.fun:.4f}")
        print(f"Gradient at the solution: {gradient}")
