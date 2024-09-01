import numpy as np
from scipy.optimize import minimize

g = 9.81

def objective(x, I, m):
    v_x0, v_y0, theta_0 = x
    T = 2 * v_y0 / g
    omega = (2 * np.pi - theta_0) / T
    v_f = np.sqrt(v_x0**2 + (3 * v_y0)**2)
    potential_energy = m * v_y0**2 / 2
    return np.sqrt(I * omega**2 + m * v_f**2 + potential_energy)

def objective_with_regularization(x, I, m):
    regularization = 1e-6 * np.sum(x**2)
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

def random_inertia():
    a = np.random.uniform(0.05, 0.2)
    b = np.random.uniform(0.05, 0.2)
    m = np.random.uniform(0.5, 2.0)
    I = (1/12) * m * (a**2 + b**2)
    return I, m, a, b

x0_list = [
    [5, 5, np.pi / 2],
    [1, 1, np.pi / 4],
    [10, 3, np.pi / 3],
    [7, 2, np.pi / 6]
]

for _ in range(4):
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
