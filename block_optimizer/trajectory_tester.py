import numpy as np
from scipy.optimize import minimize

g = 9.81

# Objective function (calculates objective value based on guess values)
def objective(x, I, m):
    v_x0, v_y0, theta_0, omega, h = x
    T = np.sqrt(2 * h / g)  
    v_f = np.sqrt(v_x0**2 + v_y0**2)  
    potential_energy = m * g * h 
    return np.sqrt(I * omega**2 + m * v_f**2 + potential_energy)

# Objective with regularization (includes regularization term)
def objective_with_regularization(x, I, m):
    x = np.array(x) 
    regularization = 1e-6 * np.sum(x**2) 
    return objective(x, I, m) + regularization

# Constraint to check if the block lands upright
def constraint_theta_final(x):
    v_x0, v_y0, theta_0, omega, h = x
    T = np.sqrt(2 * h / g)  
    return np.pi - (theta_0 + omega * T)  

# Constraint to limit angular velocity
def constraint_omega_limit(x):
    omega = x[3]
    return (2 * np.pi / 3) - omega  

# Constraint to make sure height is reasonable
def constraint_height_limit(x):
    h = x[4]
    return h - 0.1  
# Bounds for the optimization variables
bnds = [(0, None), (0, None), (0, 2 * np.pi / 3), (0, 2 * np.pi / 3), (0.1, None)]  

# Random inertia generation for each block
def random_inertia():
    a = np.random.uniform(0.05, 0.2)
    b = np.random.uniform(0.05, 0.2)
    m = np.random.uniform(0.5, 2.0)
    I = (1 / 12) * m * (a**2 + b**2)
    return I, m, a, b

# Function to evaluate objective and constraints for given guesses
def evaluate_guess(x, I, m):
    v_x0, v_y0, theta_0, omega, h = x

    # Evaluate objective function
    obj_value = objective_with_regularization(x, I, m)
    
    # Check constraints
    theta_constraint = constraint_theta_final(x)
    omega_constraint = constraint_omega_limit(x)
    height_constraint = constraint_height_limit(x)

    print("===========================")
    print(f"Guessed values: v_x0 = {v_x0:.4f}, v_y0 = {v_y0:.4f}, theta_0 = {theta_0:.4f}, omega = {omega:.4f}, h = {h:.4f}")
    print(f"Objective value: {obj_value:.4f}")

    # Checking constraint satisfaction
    print(f"Constraint: theta_final (should be 0): {theta_constraint:.4f}")
    print(f"Constraint: omega limit (should be >= 0): {omega_constraint:.4f}")
    print(f"Constraint: height limit (should be >= 0): {height_constraint:.4f}")

    if np.isclose(theta_constraint, 0, atol=1e-2):
        print("The block will land upright (theta_f = pi).")
    else:
        print("The block will NOT land upright.")
    
    if omega_constraint >= 0:
        print("The angular velocity constraint is satisfied.")
    else:
        print("The angular velocity constraint is NOT satisfied.")
    
    if height_constraint >= 0:
        print("The height constraint is satisfied.")
    else:
        print("The height constraint is NOT satisfied.")

# Input guesses for the variables
while True:
    # Input the guesses from the user
    try:
        v_x0 = float(input("Enter v_x0 (initial horizontal velocity): "))
        v_y0 = float(input("Enter v_y0 (initial vertical velocity): "))
        theta_0 = float(input("Enter theta_0 (initial angle in radians): "))
        omega = float(input("Enter omega (initial angular velocity): "))
        h = float(input("Enter h (initial height): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        continue

    # Combine input values into a guess array
    x_guess = [v_x0, v_y0, theta_0, omega, h]

    # Generate random inertia values for the block
    I, m, a, b = random_inertia()
    print(f"\nRandomized dimensions and mass: a = {a:.3f}m, b = {b:.3f}m, mass = {m:.3f}kg, inertia = {I:.5f} kg·m²")

    # Evaluate the guess with the current inertia and mass
    evaluate_guess(x_guess, I, m)

    # Option to repeat or exit
    cont = input("Do you want to input another guess? (yes/no): ").lower()
    if cont != "yes":
        break

