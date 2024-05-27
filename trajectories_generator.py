import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import scipy.optimize as opt
import pickle
import os
import cvxpy as cp
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import BFGS
from kuka_dynamics import *
from utils import *
from kuka_block_assembly import *
from scipy.interpolate import interp1d

INITIAL_STATE = 0 
INITIAL_GRASP_BLOCK = 1
LIFTING_BLOCK = 2
ASSEMBLY = 3
RELEASE_BLOCK = 4
REVERSE_TRAJECTORY = 5
GRASP_BLOCK = 6
CALIBRATE = 7 
POST_ASSEMBLY = 8
APPROACH_GRASP_BLOCK = 9
APPROACH_RELEASE = 10


def generate_initial_trajectory(pb_client, robot_id, end_effector_link_id, block_id, state, current_level, side, N_waypoints=1500):

    joint_angles_init = np.array(get_joint_angles(pb_client, robot_id))
    if side < 2:
        lifting_height = (current_level*0.03) + 0.035
    elif side == 2:
        lifting_height = (current_level*0.03) + 0.05
    else:
        lifting_height = (current_level*0.03) + 0.07

    if state == INITIAL_STATE:
        trajectory_file = 'trajectories/initial_trajectory_final.pkl'
        min_velocity = 1.7

        if os.path.exists(trajectory_file):
            print("Loading existing optimization results...")
            joint_angles_target, joint_velocities_target = load_data_with_pickle(trajectory_file)
        
        else:
            print("Running optimization process...")
            
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            start_position, _ = get_end_effector_pose(pb_client, robot_id)
            end_position = block_position + np.array((0, 0, .05))
            waypoints = basic_optimization(start_position, end_position, N_waypoints, side)
            print("Basic optimization waypoints: ", len(waypoints))
            print("Type of waypoints: ", type(waypoints))
            print("waypoints: ", waypoints)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, (0, 0, 0))[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, N_waypoints, state, current_level, side)
    
    elif state == INITIAL_GRASP_BLOCK:
        trajectory_file = 'trajectories/initial_grasp_trajectory.pkl'
        min_velocity = 0.1

        if os.path.exists(trajectory_file):
            print("Loading existing optimization results...")
            joint_angles_target, joint_velocities_target = load_data_with_pickle(trajectory_file)

        else:
            print("Running optimization process...")
            num_waypoints = 500
            min_velocity = 0.1
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            start_position, _ = get_end_effector_pose(pb_client, robot_id)
            end_position = block_position
            waypoints = basic_optimization(start_position, end_position, num_waypoints, state, current_level, side)
            print("Basic optimization waypoints: ", len(waypoints))
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, pb.getEulerFromQuaternion(block_orientation_quaternion))[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state, current_level, side)

    elif state == LIFTING_BLOCK:
        if side < 2:
            print("Running optimization process...")
            trajectory_file = f'trajectories_new/lifting_{current_level}_trajectory.pkl'
            min_velocity = 1.5
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            block_target_orientation = pb.getEulerFromQuaternion(block_orientation_quaternion)

        elif side == 2:
            print("Running optimization process...")
            trajectory_file = f'trajectories_new/c/lifting_{current_level}_trajectory.pkl'
            min_velocity = 1
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            block_target_orientation = pb.getEulerFromQuaternion(block_orientation_quaternion)
        
        else:
            print("Running optimization process...")
            trajectory_file = f'trajectories_new/stand/lifting_{current_level}_trajectory.pkl'
            min_velocity = 1
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            block_target_orientation = pb.getEulerFromQuaternion(block_orientation_quaternion)


        if os.path.exists(trajectory_file):
            print("Loading existing optimization results...")
            
            waypoints = load_data_with_pickle(trajectory_file)
            print("Last waypoint of lifting block: ", waypoints[-1])
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0


        else:
            print("Running optimization process...")
            num_waypoints = 500
            min_velocity = 1.5
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            start_position, _ = get_end_effector_pose(pb_client, robot_id)
            end_position = np.array((.5, 0, 0)) +  np.array((0, 0, lifting_height))
            print("Start position: ", start_position, "End position: ", end_position)
            waypoints = basic_optimization(start_position, end_position, num_waypoints, state, current_level, side)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0
            #joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state)


    elif state == CALIBRATE:
        min_velocity = 3
        h = 1/240
        N = 1000
        end_position = np.zeros(7)
        angles = np.linspace(joint_angles_init, end_position, N+1, axis=0)
        
        velocities = np.diff(angles, axis=0) / h
        velocities = np.vstack([velocities, np.zeros((1, 7))])
        
        joint_angles_target = angles
        joint_velocities_target = velocities


    elif state == ASSEMBLY: 
        if side < 3:    
            if side == 0:
                #print("Current level: ", current_level)
                trajectory_file = f'trajectories_new/l/assembly_trajectory_{current_level}.pkl'
                min_velocity = 1.7
                block_target_orientation = (np.pi, 0, np.pi/2)
                num_waypoints = 1500
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_l_new.pkl'

            elif side == 1:
                #print("Current level: ", current_level)
                trajectory_file = f'trajectories_new/r/assembly_trajectory_{current_level}.pkl'
                min_velocity = 1.7
                block_target_orientation = (np.pi, 0, np.pi/2)
                num_waypoints = 1500
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_r_new.pkl'

            elif side == 2:
                #print("Current level: ", current_level)
                trajectory_file = f'trajectories_new/c/assembly_trajectory_{current_level}.pkl'
                min_velocity = 1.7
                block_target_orientation = (np.pi, 0, np.pi)
                num_waypoints = 1500
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_c_new.pkl'

            #elif side == 3:
                #print("Current level: ", current_level)
                #trajectory_file = f'trajectories_new/stand/assembly_trajectory_{current_level}.pkl'
                #min_velocity = 1
                #block_target_orientation = (np.pi, np.pi/2, 0)
                #num_waypoints = 1500
                #bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_r_new.pkl'


            if os.path.exists(trajectory_file):
                #print("Loading existing optimization results...")
                joint_angles_target, joint_velocities_target = load_data_with_pickle(trajectory_file)
                #print("First waypoint of assembly: ", waypoints[0])

            else:
                data = load_data_with_pickle(bspline_file)
                p_x = data['p_x']
                p_y = data['p_y']
                p_x = np.array(p_x)
                p_y = np.array(p_y)

                p_x = p_x - 3.95
                p_y = p_y - 3.95
                p_x = p_x * 0.2
                p_y = p_y * 0.2

                t = np.linspace(0, 1, len(p_x))
                interp_func_x = interp1d(t, p_x, kind='cubic')
                interp_func_y = interp1d(t, p_y, kind='cubic')
                t_new = np.linspace(0, 1, num_waypoints)  

                dense_pts_x = interp_func_x(t_new)
                dense_pts_y = interp_func_y(t_new)

                p_x = dense_pts_x
                p_y = dense_pts_y

                p_z = lifting_height * np.ones(len(p_x))
                
                waypoints = np.array([np.array([x, y, z]) for x, y, z in zip(p_x, p_y, p_z)])
                num_waypoints = len(waypoints)

                theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
                theta_waypoints_array = np.array(theta_waypoints)
                #joint_angles_target= theta_waypoints_array
                #joint_velocities_target = 0
                #if side == 2:
                #    joint_angles_target, joint_velocities_target = feasibility_optimization_simpler(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state, current_level, side)
                #    print("Ini ketrigger")
                if side < 3:
                    joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state, current_level, side)
                else:
                    joint_angles_target, joint_velocities_target = feasibility_optimization_simpler(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state, current_level, side)
        else:
            h = 1/240
            N = 5000
            min_velocity = 0.5
            block_target_orientation = (0, np.pi/2, 0)
            end_position = inverse_kinematics(pb_client, robot_id, end_effector_link_id, (-.06, -.82, (lifting_height) + 0.05), block_target_orientation)[:7]
            angles = np.linspace(joint_angles_init, end_position, N+1, axis=0)
            
            velocities = np.diff(angles, axis=0) / h
            velocities = np.vstack([velocities, np.zeros((1, 7))])
            
            joint_angles_target = angles
            joint_velocities_target = velocities
  
    elif state == POST_ASSEMBLY:
        h = 1/240
        N = 1000
        min_velocity = 1.9
        block_target_orientation = (np.pi, np.pi/2, 0)
        end_position = inverse_kinematics(pb_client, robot_id, end_effector_link_id, (-1, 0, lifting_height + .5), block_target_orientation)[:7]
        angles = np.linspace(joint_angles_init, end_position, N+1, axis=0)
        
        velocities = np.diff(angles, axis=0) / h
        velocities = np.vstack([velocities, np.zeros((1, 7))])
        
        joint_angles_target = angles
        joint_velocities_target = velocities
        
        #print("Running optimization process...")
        #trajectory_file = f'trajectories_new/stand/postassembly_{current_level}_trajectory.pkl'
        #min_velocity = 1
        #block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
        #block_target_orientation = (np.pi/2, np.pi/2, np.pi)


        #if os.path.exists(trajectory_file):
        #    print("Loading existing optimization results...")
        #    waypoints = load_data_with_pickle(trajectory_file)
        #    print("Last waypoint of lifting block: ", waypoints[-1])
        #    theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
        #    theta_waypoints_array = np.array(theta_waypoints)
        #    joint_angles_target= theta_waypoints_array
        #    joint_velocities_target = 0


        #else:
        #    print("Running optimization process...")
        #    num_waypoints = 500
        #    min_velocity = 1.5
        #    block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
        #    start_position, _ = get_end_effector_pose(pb_client, robot_id)
        #    end_position = np.array((-.09, -.60, lifting_height))
        #    print("Start position: ", start_position, "End position: ", end_position)
        #    waypoints = basic_optimization(start_position, end_position, num_waypoints, state, current_level, side)
        #    theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
        #    theta_waypoints_array = np.array(theta_waypoints)
        #    joint_angles_target= theta_waypoints_array
        #    joint_velocities_target = 0
            #joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state)

    elif state == APPROACH_RELEASE:
        h = 1/240
        N = 20000
        min_velocity = 1.25
        if side == 0:
            block_target_orientation = (np.pi/2, 0, np.pi/2)
            block_target_position = (-.13, -.79, (lifting_height)+0.15)
        elif side == 1:
            block_target_orientation = (np.pi/2, 0, np.pi/2)
            block_target_position = (-.05, -.79, (lifting_height)+0.15)
        elif side == 2:
            if current_level == 11:
                block_target_orientation = (np.pi, 0, np.pi)
                block_target_position = (-.05, -.74, (lifting_height) + 0.25)
            else:
                block_target_orientation = (np.pi, 0, np.pi)
                block_target_position = (-.05, -.80, (lifting_height) + 0.25)
        else:
            block_target_orientation = (np.pi, np.pi/2, 0)
            block_target_position = (-.07, -.84, (lifting_height)+0.12)    
        end_position = inverse_kinematics(pb_client, robot_id, end_effector_link_id, block_target_position, block_target_orientation)[:7]
        angles = np.linspace(joint_angles_init, end_position, N+1, axis=0)
        
        velocities = np.diff(angles, axis=0) / h
        velocities = np.vstack([velocities, np.zeros((1, 7))])
        
        joint_angles_target = angles
        joint_velocities_target = velocities

    elif state == RELEASE_BLOCK:
        if side == 0:
            trajectory_file = f'trajectories_new/l/releasing_{current_level}_trajectory.pkl'
            min_velocity = 0.1
            end_position = (-.13, -.79, (lifting_height))
            block_target_orientation = (np.pi, 0, np.pi/2)

        elif side == 1:
            trajectory_file = f'trajectories_new/r/releasing_{current_level}_trajectory.pkl'
            min_velocity = 0.1
            end_position = (-.05, -.79, (lifting_height))
            block_target_orientation = (np.pi, 0, np.pi/2)

        elif side == 2:
            trajectory_file = f'trajectories_new/c/releasing_{current_level}_trajectory.pkl'
            min_velocity = 0
            end_position = (-.09, -.79, (lifting_height))
            block_target_orientation = (np.pi, 0, np.pi)

        else:
            trajectory_file = f'trajectories_new/stand/releasing_{current_level}_trajectory.pkl'
            min_velocity = 0.1
            end_position = (-.10, -.88, lifting_height+0.03)
            block_target_orientation = (0, np.pi/2, 0)
        #elif side == 3:
            #trajectory_file = f'trajectories/d/releasing_{current_level}_trajectory.pkl'
            #min_velocity = 0.25
            #end_position = (-.11, -.51, (lifting_height))
            #block_target_orientation = (np.pi, 0, np.pi)

        if os.path.exists(trajectory_file):
            print("Loading existing optimization results...")
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            waypoints = load_data_with_pickle(trajectory_file)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0
        
        else:
            print("Running optimization process...")
            if side == 2:
                num_waypoints = 1500
            else:
                num_waypoints = 500
            start_position, _ = get_end_effector_pose(pb_client, robot_id)
            print("Start position: ", start_position, "End position: ", end_position)
            waypoints = basic_optimization(start_position, end_position, num_waypoints, state, current_level, side)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, block_target_orientation)[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0
            #joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state)

 
    elif state == REVERSE_TRAJECTORY:
        

        if side == 0:
            print("Current level: ", current_level)
            trajectory_file = f'trajectories_new/l/reverse_trajectory_{current_level}.pkl'
            min_velocity = 1.7
            bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_r_new.pkl'

        elif side == 1:
            print("Current level: ", current_level)
            trajectory_file = f'trajectories_new/r/reverse_trajectory_{current_level}.pkl'
            min_velocity = 1.7
            bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_l_new.pkl'

        elif side == 2:
            print("Current level: ", current_level)
            trajectory_file = f'trajectories_new/c/reverse_trajectory_{current_level}.pkl'
            min_velocity = 1
            if current_level == 7 or current_level == 1:
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_r_new.pkl'
            else:
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_c_new.pkl'

        else:
            print("Current level: ", current_level)
            trajectory_file = f'trajectories_new/stand/reverse_trajectory_{current_level}.pkl'
            min_velocity = 1
            if current_level == 14 or current_level == 1:
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_r_new.pkl'
            else:
                bspline_file = 'bspline_trajectory_optimization_edt/grasp_pick_trajectory_c_new.pkl'


        if os.path.exists(trajectory_file):
            print("Loading existing optimization results: ", trajectory_file)
            joint_angles_target, joint_velocities_target = load_data_with_pickle(trajectory_file)

        else:
            data = load_data_with_pickle(bspline_file)
            p_x = data['p_x']
            p_y = data['p_y']
            p_x = np.array(p_x)
            p_y = np.array(p_y)

            p_x = p_x - 3.95
            p_y = p_y - 3.95
            p_x = p_x * 0.2
            p_y = p_y * 0.2

            t = np.linspace(0, 1, len(p_x))
            interp_func_x = interp1d(t, p_x, kind='cubic')
            interp_func_y = interp1d(t, p_y, kind='cubic')
            t_new = np.linspace(0, 1, 1500)  

            dense_pts_x = interp_func_x(t_new)
            dense_pts_y = interp_func_y(t_new)

            p_x = np.flip(dense_pts_x)
            p_y = np.flip(dense_pts_y)

            p_z = lifting_height * np.ones(len(p_x))
            
            waypoints = np.array([np.array([x, y, z]) for x, y, z in zip(p_x, p_y, p_z)])
            num_waypoints = len(waypoints)

            print("Joint angles when placing block: ", joint_angles_init)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, (np.pi, 0, np.pi))[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            #joint_angles_target= theta_waypoints_array
            #joint_velocities_target = 0
            joint_angles_target, joint_velocities_target = feasibility_optimization_simpler(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state, current_level, side)

            

    elif state == GRASP_BLOCK:
        if side == 0:
            trajectory_file = f'trajectories_new/l/grasping_{current_level}_trajectory.pkl'
            min_velocity = 0.4

        elif side == 1:
            trajectory_file = f'trajectories_new/r/grasping_{current_level}_trajectory.pkl'
            min_velocity = 0.4

        elif side == 2:
            trajectory_file = f'trajectories_new/c/grasping_{current_level}_trajectory.pkl'
            min_velocity = 0.25

        else:
            trajectory_file = f'trajectories_new/stand/grasping_{current_level}_trajectory.pkl'
            min_velocity = 0.35
        #elif side == 3:
            #trajectory_file = f'trajectories/d/grasping_{current_level}_trajectory.pkl'
            #min_velocity = 0.25


        if os.path.exists(trajectory_file):
            print("Loading existing optimization results: ", trajectory_file)
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            waypoints = load_data_with_pickle(trajectory_file)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, pb.getEulerFromQuaternion(block_orientation_quaternion))[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0
        
        else:
            print("Running optimization process...")
            num_waypoints = 2000
            block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)
            start_position, _ = get_end_effector_pose(pb_client, robot_id)
            end_position = block_position
            print("Start position: ", start_position, "End position: ", end_position)
            waypoints = basic_optimization(start_position, end_position, num_waypoints, state, current_level, side)
            theta_waypoints = [inverse_kinematics(pb_client, robot_id, end_effector_link_id, waypoint, pb.getEulerFromQuaternion(block_orientation_quaternion))[:7] for waypoint in waypoints]
            theta_waypoints_array = np.array(theta_waypoints)
            joint_angles_target= theta_waypoints_array
            joint_velocities_target = 0
            #joint_angles_target, joint_velocities_target = feasibility_optimization(pb_client, robot_id, end_effector_link_id, joint_angles_init, theta_waypoints_array, num_waypoints, state)

    elif state == APPROACH_GRASP_BLOCK:
        h = 1/240
        N = 5000
        min_velocity = 1.25
        block_position, block_orientation_quaternion = get_block_pose(pb_client, block_id)  
        block_target_position = block_position + np.array((0, 0, .05))
        end_position = inverse_kinematics(pb_client, robot_id, end_effector_link_id, block_target_position, pb.getEulerFromQuaternion(block_orientation_quaternion))[:7]
        angles = np.linspace(joint_angles_init, end_position, N+1, axis=0)
        
        velocities = np.diff(angles, axis=0) / h
        velocities = np.vstack([velocities, np.zeros((1, 7))])
        
        joint_angles_target = angles
        joint_velocities_target = velocities

    return joint_angles_target, joint_velocities_target, min_velocity

    

def feasibility_optimization(pb_client, robot_id, end_effector_link_id, current_theta, theta_ref, N, state, current_level, current_side):

    max_linear_velocity = 1.7  
    h = 1/240  

    theta_min = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543])
    theta_max = np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    theta_min_tiled = np.tile(theta_min, (N, 1))
    theta_max_tiled = np.tile(theta_max, (N, 1))

    theta = cp.Variable((N, 7))
    theta_dot = cp.Variable((N-1, 7))

    cost = cp.sum_squares(theta - theta_ref) + cp.sum_squares(theta_dot)

    constraints = [theta >= theta_min_tiled, theta <= theta_max_tiled]
    constraints += [theta[0, :] == current_theta, theta[-1, :] == theta_ref[-1, :]]
    #constraints += [theta[0, :] == current_theta]

    for i in range(N-1):
        constraints.append(theta_dot[i, :] == (theta[i+1, :] - theta[i]) / h)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    solver_options = {
    'max_iter': 10000,
    'eps_abs': 1e-3,
    'eps_rel': 1e-3}

    problem.solve(solver=cp.OSQP, **solver_options, verbose=True)
    print("First optimization status: ", problem.status)

    # Extract solution for further processing
    theta_solution = theta.value

    for i in range(N-1):
        theta_i = theta_solution[i, :]
        J = compute_jacobian(pb_client, robot_id, end_effector_link_id, theta_i)  

        # Compute velocity
        v = J @ theta_dot[i, :]
        constraints.append(cp.norm(v, 2) <= max_linear_velocity)

    # Redefine and resolve the problem with updated velocity constraints
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(max_iters=10000)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        print("Final Optimization Successful")
        if state == INITIAL_STATE:
            save_data_with_pickle((theta.value, theta_dot.value), 'trajectories/initial_trajectory.pkl')
        elif state == INITIAL_GRASP_BLOCK:
            save_data_with_pickle((theta.value, theta_dot.value), 'trajectories/initial_grasp_trajectory.pkl')
        elif state == ASSEMBLY:
            if current_side == 0:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/l/assembly_trajectory_{current_level}.pkl')
            elif current_side == 1:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/r/assembly_trajectory_{current_level}.pkl')
            elif current_side == 2:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/c/assembly_trajectory_{current_level}.pkl')
            else:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/stand/assembly_trajectory_{current_level}.pkl')

        return theta.value, theta_dot.value
    else:
        print("Final Optimization Failed:", problem.status)
        return None, None
    
def feasibility_optimization_simpler(pb_client, robot_id, end_effector_link_id, current_theta, theta_ref, N, state, current_level, current_side):

    max_linear_velocity = 1.7  # m/s
    #min_linear_velocity = 1
    h = 1/240  

    theta_min = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543])
    theta_max = np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    theta_min_tiled = np.tile(theta_min, (N, 1))
    theta_max_tiled = np.tile(theta_max, (N, 1))

    theta = cp.Variable((N, 7))
    theta_dot = cp.Variable((N-1, 7))

    cost = cp.sum_squares(theta - theta_ref) + cp.sum_squares(theta_dot)

    constraints = [theta >= theta_min_tiled, theta <= theta_max_tiled]
    #constraints += [theta[0, :] == current_theta, theta[-1, :] == theta_ref[-1, :]]
    constraints += [theta[0, :] == current_theta]

    for i in range(N-1):
        constraints.append(theta_dot[i, :] == (theta[i+1, :] - theta[i]) / h)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    solver_options = {
    'max_iter': 10000,
    'eps_abs': 1e-3,
    'eps_rel': 1e-3}

    problem.solve(solver=cp.OSQP, **solver_options, verbose=True)
    print("First optimization status: ", problem.status)

    # Extract solution for further processing
    theta_solution = theta.value

    for i in range(N-1):
        theta_i = theta_solution[i, :]
        J = compute_jacobian(pb_client, robot_id, end_effector_link_id, theta_i)  

        # Compute velocity
        v = J @ theta_dot[i, :]
        constraints.append(cp.norm(v, 2) <= max_linear_velocity)

    # Redefine and resolve the problem with updated velocity constraints
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(max_iters=10000)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        print("Final Optimization Successful")
        if state == REVERSE_TRAJECTORY:
            if current_side == 0:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/l/reverse_trajectory_{current_level}.pkl')
            elif current_side == 1:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/r/reverse_trajectory_{current_level}.pkl')
            elif current_side == 2:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/c/reverse_trajectory_{current_level}.pkl')
            else:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/stand/reverse_trajectory_{current_level}.pkl')
        elif state == ASSEMBLY:
            if current_side == 0:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/l/assembly_trajectory_{current_level}.pkl')
            elif current_side == 1:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/r/assembly_trajectory_{current_level}.pkl')
            elif current_side == 2:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/c/assembly_trajectory_{current_level}.pkl')
            else:
                save_data_with_pickle((theta.value, theta_dot.value), f'trajectories_new/stand/assembly_trajectory_{current_level}.pkl')
        return theta.value, theta_dot.value
    else:
        print("Final Optimization Failed:", problem.status)
        return None, None
    



def basic_optimization(start, end, num_waypoints, state, current_level, current_side, max_distance=0.2 ):
    waypoints = cp.Variable((num_waypoints, 3))
    objective = cp.Minimize(sum(cp.norm(waypoints[i] - waypoints[i-1], 2) for i in range(1, num_waypoints)))

    constraints = [waypoints[0] == start, waypoints[-1] == end]  # Start and end points
    constraints += [cp.norm(waypoints[i] - waypoints[i-1], 2) <= max_distance for i in range(1, num_waypoints)]  # Max distance

    problem = cp.Problem(objective, constraints)
    problem.solve()

    optimized_waypoints = waypoints.value

    if problem.status in ["optimal", "optimal_inaccurate"]:
        print("Final Optimization Successful")
        if state == LIFTING_BLOCK:
            if current_side < 2:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/lifting_{current_level}_trajectory.pkl')
            elif current_side == 2:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/c/lifting_{current_level}_trajectory.pkl')
            else:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/stand/lifting_{current_level}_trajectory.pkl')
        elif state == RELEASE_BLOCK:
            if current_side == 0:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/l/releasing_{current_level}_trajectory.pkl')
            elif current_side == 1:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/r/releasing_{current_level}_trajectory.pkl')
            elif current_side == 2:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/c/releasing_{current_level}_trajectory.pkl')            
            else:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/stand/releasing_{current_level}_trajectory.pkl')            
            
        elif state == GRASP_BLOCK:
            if current_side == 0:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/l/grasping_{current_level}_trajectory.pkl')
            elif current_side == 1:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/r/grasping_{current_level}_trajectory.pkl')
            elif current_side == 2:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/c/grasping_{current_level}_trajectory.pkl')
            else:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/stand/grasping_{current_level}_trajectory.pkl')
        
        elif state == ASSEMBLY:
            if current_side == 2:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/c/assembly_trajectory_{current_level}.pkl')
            elif current_side == 3:
                save_data_with_pickle((optimized_waypoints), f'trajectories_new/stand/assembly_trajectory_{current_level}.pkl')

        elif state == POST_ASSEMBLY:
            save_data_with_pickle((optimized_waypoints), f'trajectories_new/stand/postassembly_trajectory_{current_level}.pkl')
        return optimized_waypoints

TIME_STEP = 1 / 240  

def generate_final_trajectory(pb_client, robot_id, end_effector_link_id, joint_angles_target, min_velocity, max_velocity=1.8):
    
    waypoint_index = 0
    processed_joint_angles_target = [joint_angles_target[0]]
    accepted_waypoints = []  # List to store data about rejected waypoints
    
    while waypoint_index < len(joint_angles_target) - 1:
        current_joint_angles = processed_joint_angles_target[-1]
        next_joint_angles = joint_angles_target[waypoint_index + 1]
        
        try:
            linear_jacobian = compute_jacobian(pb_client, robot_id, end_effector_link_id, next_joint_angles)
            joint_velocities = (next_joint_angles - current_joint_angles) / TIME_STEP
            next_velocity = np.linalg.norm(linear_jacobian @ joint_velocities)

            if waypoint_index == len(joint_angles_target) - 2:  # Check if it's the second-to-last waypoint
                processed_joint_angles_target.append(joint_angles_target[-1])  # Ensure last point is always included
                accepted_waypoints.append((waypoint_index + 1, next_velocity))
            elif next_velocity >= max_velocity:
                interpolated_joint_angles = interpolate_joint_angles(current_joint_angles, next_joint_angles, max_velocity, linear_jacobian)
                processed_joint_angles_target.append(interpolated_joint_angles)
                accepted_waypoints.append((waypoint_index + 1, next_velocity))
            elif min_velocity <= next_velocity < max_velocity:
                processed_joint_angles_target.append(next_joint_angles)
                accepted_waypoints.append((waypoint_index + 1, next_velocity))
            #else:
                #print(f"Waypoint {waypoint_index + 1} excluded, velocity: {next_velocity}")

        except Exception as e:
            print(f"Error computing Jacobian for waypoint {waypoint_index + 1}: {e}")

        waypoint_index += 1

    joint_angles_target = np.array(processed_joint_angles_target)
    joint_velocities_target = np.diff(joint_angles_target, axis=0) / TIME_STEP
    joint_velocities_target = np.vstack([joint_velocities_target, np.zeros(joint_angles_target.shape[1])])
    
    joint_accelerations_target = np.diff(joint_velocities_target, axis=0) / TIME_STEP
    joint_accelerations_target = np.vstack([joint_accelerations_target, np.zeros(joint_angles_target.shape[1])])

    return joint_angles_target, joint_velocities_target, joint_accelerations_target, accepted_waypoints

def interpolate_joint_angles(current_angles, next_angles, max_velocity, jacobian):
    low = 0
    high = 1
    while high - low > 1e-6:  # tolerance for interpolation
        mid = (low + high) / 2
        test_angles = current_angles + mid * (next_angles - current_angles)
        test_velocities = (test_angles - current_angles) / TIME_STEP
        test_velocity = np.linalg.norm(jacobian @ test_velocities)

        if test_velocity > max_velocity:
            high = mid
        else:
            low = mid

    return current_angles + low * (next_angles - current_angles)


