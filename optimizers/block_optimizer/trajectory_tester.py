import numpy as np
from scipy.optimize import minimize

g = 9.81

# Objective function (calculates objective value based on guess values)
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
    obj_value = objective(x, I, m)
    
    # Check constraints
    theta_constraint = constraint_theta_final(x)
    omega_constraint = constraint_omega_limit(x)

    print("===========================")
    print(f"Guessed values: v_x0 = {v_x0:.4f}, v_y0 = {v_y0:.4f}, theta_0 = {theta_0:.4f}, omega = {omega:.4f}, h = {h:.4f}")
    print(f"Objective value: {obj_value:.4f}")

    # Checking constraint satisfaction
    print(f"Constraint: theta_final (should be 0): {theta_constraint:.4f}")
    print(f"Constraint: omega limit (should be >= 0): {omega_constraint:.4f}")

    if np.isclose(theta_constraint, 0, atol=5e-2):
        print("The block will land upright (theta_f = pi).")
    else:
        print("The block will NOT land upright.")
    
    if omega_constraint >= 0:
        print("The angular velocity constraint is satisfied.")
    else:
        print("The angular velocity constraint is NOT satisfied.")
    


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

