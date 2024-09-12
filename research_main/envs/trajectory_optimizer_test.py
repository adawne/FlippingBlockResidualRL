import numpy as np
from scipy.optimize import minimize
import argparse

g = 9.81

def random_inertia():
    a = np.random.uniform(0.05, 0.2)
    b = np.random.uniform(0.05, 0.2)
    m = np.random.uniform(0.5, 2.0)
    I = (1/12) * m * (a**2 + b**2)
    return I, m, a, b

def calculate_inertia(m, a, b):
    return (1/12) * m * (a**2 + b**2)

def objective(x, I, m, lambdas):
    v_x0, v_y0, theta_0, h_release = x
    
    h_max = h_release + (v_y0**2) / (2 * g)
    T = v_y0 / g + np.sqrt(2 * h_max / g)
    v_landing = np.sqrt(v_x0**2 + (g*T)**2)
    omega_landing = (2 * np.pi - theta_0) / T
    
    translational_energy = lambdas[0] * 0.5 * m * v_landing**2
    rotational_energy = lambdas[1] * 0.5 * I * omega_landing**2
    potential_energy = lambdas[2] * m * g * h_max
    
    return translational_energy + rotational_energy + potential_energy

def objective_with_regularization(x, I, m, lambdas):
    regularization = 1e-6 * np.sum(x**2)
    return objective(x, I, m, lambdas) + regularization

def constraint_omega_lower(x):
    v_x0, v_y0, theta_0, h_release = x
    T = v_y0 / g + np.sqrt(2 * (h_release + (v_y0**2) / (2 * g)) / g)
    omega = (2 * np.pi - theta_0) / T
    return omega + np.pi  

def constraint_omega_upper(x):
    v_x0, v_y0, theta_0, h_release = x
    T = v_y0 / g + np.sqrt(2 * (h_release + (v_y0**2) / (2 * g)) / g)
    omega = (2 * np.pi - theta_0) / T
    return np.pi - omega  

def generate_solution(I, m, a, b, lambdas):
    print(f"\nUsing dimensions and mass: a = {a:.3f}m, b = {b:.3f}m, mass = {m:.3f}kg, inertia = {I:.5f} kg·m²")
    
    bnds = [(0.1, None), (0.1, 2), (0, 2*np.pi/3), (0.46, None)]
    x0_list = [
        [5, 5, np.pi / 2, 1]
        #[1, 1, np.pi / 4, 0.5],
        #[10, 3, np.pi / 3, 2],
        #[7, 2, np.pi / 6, 1.5]
    ]
    
    constr = [{'type': 'ineq', 'fun': constraint_omega_lower}, {'type': 'ineq', 'fun': constraint_omega_upper}]
    
    for x0 in x0_list:
        result = minimize(objective_with_regularization, x0, args=(I, m, lambdas), method='trust-constr', bounds=bnds, constraints=constr)
        v_x0_opt, v_y0_opt, theta_0_opt, h_release_opt = result.x
        
        T_opt = v_y0_opt / g + np.sqrt(2 * (h_release_opt + (v_y0_opt**2) / (2 * g)) / g)
        omega_opt = (0 - theta_0_opt) / T_opt
        
        gradient = result.jac
        lagrange_multipliers_constraints = result.v[0:2]
        lagrange_multipliers_bounds = result.v[2:]
        
        print("===========================")
        print(f"Initial guess: {x0}")
        print(f"Optimized values: v_x0 = {v_x0_opt:.4f}, v_y0 = {v_y0_opt:.4f}, theta_0 = {theta_0_opt:.4f}, h_release = {h_release_opt:.4f}")
        print(f"Objective value: {result.fun:.4f}")
        print(f"Time in the air: {T_opt:.4f} and Omega at landing: {omega_opt:.4f} rad/s")
        print(f"Gradient at the solution: {gradient}")
        
        # Lagrange multipliers (dual variables)
        print(f"Lagrange multipliers for omega constraints: {lagrange_multipliers_constraints}")
        print(f"Lagrange multipliers for bounds (including h_release): {lagrange_multipliers_bounds}")
        print("===========================")

def evaluate_parameters(v_x0, v_y0, theta_0, h_release, I, m, lambdas):
    x = [v_x0, v_y0, theta_0, h_release]
    obj_value = objective(x, I, m, lambdas)
    omega_constraint_lower = constraint_omega_lower(x)
    omega_constraint_upper = constraint_omega_upper(x)
    
    print("===========================")
    print(f"Evaluating with v_x0 = {v_x0}, v_y0 = {v_y0}, theta_0 = {theta_0}, h_release = {h_release}")
    print(f"Objective value: {obj_value:.4f}")
    
    if omega_constraint_lower >= 0 and omega_constraint_upper >= 0:
        print("Omega constraint (rot. velocity between -10 and 10 rad/s) is satisfied")
    else:
        print("Omega constraint (rot. velocity between -10 and 10 rad/s) is violated")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Flip Optimization")

    parser.add_argument("--testing_parameters", action="store_true", help="Test specific parameters instead of generating a solution")
    parser.add_argument("--v_x0", type=float, help="Initial horizontal velocity")
    parser.add_argument("--v_y0", type=float, help="Initial vertical velocity")
    parser.add_argument("--theta_0", type=float, help="Initial angle (radians)")
    parser.add_argument("--h_release", type=float, help="Release height (meters)")
    parser.add_argument("--m", type=float, help="Mass")
    parser.add_argument("--a", type=float, help="Dimension a")
    parser.add_argument("--b", type=float, help="Dimension b")
    parser.add_argument("--use_random_inertia", action="store_true", help="Use random inertia values")
    parser.add_argument("--lambdas", nargs=3, type=float, default=[1, 1, 1], help="Weights for translational, rotational, and potential energy")

    args = parser.parse_args()

    if args.testing_parameters:
        if None in [args.v_x0, args.v_y0, args.theta_0, args.h_release, args.m, args.a, args.b]:
            print("All parameters (v_x0, v_y0, theta_0, h_release, m, a, b) must be provided when testing specific parameters.")
        else:
            I = calculate_inertia(args.m, args.a, args.b)
            evaluate_parameters(args.v_x0, args.v_y0, args.theta_0, args.h_release, I, args.m, args.lambdas)
    else:
        if args.use_random_inertia:
            I, m, a, b = random_inertia()
        else:
            if None in [args.m, args.a, args.b]:
                print("Mass and dimensions (a, b) must be provided unless using random inertia.")
            else:
                I = calculate_inertia(args.m, args.a, args.b)
                m, a, b = args.m, args.a, args.b
        
        generate_solution(I, m, a, b, args.lambdas)
