import numpy as np
import mujoco

from scipy.optimize import minimize
from scene_builder import *

block_positions_orientations = [([0.5, 0.5, 0.1], [np.pi/2, 0, 0])]
world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations)
model = mujoco.MjModel.from_xml_string(world_xml_model)

#m = model.body('block_0').mass
#w = model.geom('blue_subbox').size[0]
#d = model.geom('blue_subbox').size[1]
#h = model.geom('blue_subbox').size[2]
#I = (m/12) * (w**2 + d**2)  # Moment of inertia
#print("mass: ", m, "width: ", w, "depth: ", d, "height: ", h, "manual calculation of Inertria: ", I)

#I = model.body('block_0').inertia[2]

#theta_0 = 0.0
#theta_final = 0 


# Constants
g = 9.81  # gravitational acceleration (m/s^2)

# Objective function
def objective(x):
    T, v_x0, theta_0 = x  # Now we also optimize theta_0
    omega = (2 * np.pi - theta_0) / T
    v_f = np.sqrt(v_x0**2 + (g * T / 2)**2)
    
    # Objective: minimize angular velocity + lambda * final linear velocity
    lambda_ = 0.1  # Weighting factor, adjust as needed
    return np.abs(omega) + lambda_ * v_f

# Constraints
def constraint_theta_final(x):
    T, v_x0, theta_0 = x
    omega = (2 * np.pi - theta_0) / T
    theta_final = theta_0 + omega * T
    return theta_final - 1e-5
    

# Bounds for T, v_x0, and theta_0
bnds = [(0.1, None), (0.1, None), (0, np.pi)]  # T > 0, v_x0 > 0, 0 <= theta_0 <= 2*pi

# Constraints dictionary
constr = {'type': 'eq', 'fun': lambda x: np.round(constraint_theta_final(x), decimals=5)}  # Added rounding for tolerance

x0_list = [
    [5, 5, 3 * np.pi / 2],
    [10, 1, np.pi],
    [1, 10, np.pi / 2],
    [7, 7, np.pi / 4]
]

# Loop through each initial guess and perform optimization
for x0 in x0_list:
    result = minimize(objective, x0, method='trust-constr', bounds=bnds, constraints=[constr])
    
    # Extract optimized variables
    T_opt, v_x0_opt, theta_0_opt = result.x

    # Inspect the gradient (Jacobian)
    gradient = result.jac

    # Calculate omega and vz
    omega = (2*np.pi - theta_0_opt)/T_opt
    v_z = g * T_opt/2
    
    # Output results
    print("===========================")
    print(f"Initial guess: {x0}")
    print(f"Optimized T: {T_opt}, v_x0: {v_x0_opt}, theta_0: {theta_0_opt}")
    print(f"Initial angular velocity: {omega}, initial vertical velocity: {v_z}")
    print(f"Objective value: {result.fun}")
    print(f"Gradient at the solution: {gradient}")


