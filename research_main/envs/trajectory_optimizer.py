import numpy as np
from scipy.optimize import minimize

# Parameters (example values)
I = 1.0  # Moment of inertia
m = 1.0  # Mass of the block
theta_0 = 0.0
theta_final = np.pi / 2  # 90 degrees
t = 1.0  # Time duration for the flip

# Define the objective function
def objective(x):
    omega_final = x[0]
    Vx_final = x[1]
    return np.sqrt(I * omega_final**2 + m * Vx_final**2)

# Define the constraint functions
def constraint_omega_initial(x):
    omega_final = x[0]
    return omega_final - (theta_final - theta_0) / t

def constraint_velocity(x):
    Vx_final = x[1]
    return Vx_final

# Initial guess
x0 = [0.0, 0.1]  # Initial guess for [omega_final, Vx_final]

# Define the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_omega_initial},
    {'type': 'ineq', 'fun': constraint_velocity}  # Vx_final > 0
]

# Perform the optimization
result = minimize(objective, x0, constraints=constraints, method='SLSQP')

# Output the results
print("Optimal omega_final:", result.x[0])
print("Optimal Vx_final:", result.x[1])
print("Objective function value:", result.fun)
