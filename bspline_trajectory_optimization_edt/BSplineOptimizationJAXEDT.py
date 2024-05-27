import pdb, itertools
import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
from scipy.ndimage import distance_transform_edt as edt_transform
from shapely.geometry import Polygon, Point
from jax import lax
import matplotlib.pyplot as plt

import cyipopt
import math

from jax.config import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jit, grad, jacfwd, jacrev
from functools import partial

class BSplineOptimization(cyipopt.Problem):
    def __init__(self,
                 number_of_control_points = 20,
                 dt = .2,
                 xlim = (0, 10),
                 ylim = (0, 10)):

        self.weight_velocity = 1
        self.weight_acceleration = 1
        self.weight_obstacle_avoidance = 1

        ## Width and length of the space
        self.dx = xlim[1]-xlim[0]
        self.dy = ylim[1]-ylim[0]
        
        ## Optimization parameters
        self.number_of_control_points = number_of_control_points
        self.degree_of_bspline = 3
        self.dt = dt

        ## B-Spline parameters
        self.number_of_variables = self.number_of_control_points + self.degree_of_bspline # total number of optimization variables per axis
        
        self.M_3 = .5*np.matrix([[1, 1, 0],
                                 [-2, 2, 0],
                                 [1, -2, 1]])

        MGM = np.matrix([[1 , -2, 1],
                         [-2, 4, -2],
                         [1, -2, 1]])

        self.MGM_T = np.matrix(np.zeros((self.number_of_variables, self.number_of_variables)))
        for i in range(self.number_of_control_points):
            project_matrix = np.concatenate((np.zeros((i, self.degree_of_bspline)), 
                                             np.eye(self.degree_of_bspline, self.degree_of_bspline), 
                                             np.zeros((self.number_of_control_points-i, self.degree_of_bspline))))
            self.MGM_T += project_matrix*MGM*project_matrix.T
    
        self.MGM_T = block_diag(self.MGM_T, self.MGM_T) # Make it twice bigger as there are x,y-coordinate

        self.M_3 = jnp.array(self.M_3)
        self.MGM_T = jnp.array(self.MGM_T)

        self.u_M_pos = np.dot(np.array([(1,u,u**2) for u in np.arange(0,1,self.dt)]), self.M_3)
        self.u_M_vel = np.dot(np.array([(0,1,2*u) for u in np.arange(0,1,self.dt)]), self.M_3)

        self.gradient = jit(grad(self.objective))  # objective gradient

        ## Obstacle map
        self.obstacle_map_dim = ((1000, 1000))
        self.obstacle_map = self.create_obstacle_map(self.obstacle_map_dim)
        self.edt_map = jnp.array(edt_transform(self.obstacle_map)/100+0.0001)

        plt.figure(figsize=(5,5))
        plt.imshow(self.edt_map, cmap='viridis')
        plt.show()

        plt.figure(figsize=(5,5))
        plt.imshow(self.obstacle_map, cmap='gray')
        plt.show()


    def create_obstacle_map(self, dimension):
        obstacle_map = np.ones(dimension)
        scale_factor = dimension[0] // 10 

        obstacles = [
            #[[3, 2], [4, 4]]  # Obstacle 1 corners
            #[[6, 0], [8, 1]]   # Obstacle 2 corners
        ]

        for obstacle in obstacles:
            bottom_left, top_right = obstacle
            x1, y1 = bottom_left
            x2, y2 = top_right

            # Scale the coordinates by the scale factor
            x1_scaled, x2_scaled = x1 * scale_factor, x2 * scale_factor
            y1_scaled, y2_scaled = y1 * scale_factor, y2 * scale_factor

            # Calculate image coordinates for y by inverting the y-scale to array index
            y1_image, y2_image = max(dimension[1] - y1_scaled, dimension[1] - y2_scaled), min(dimension[1] - y1_scaled, dimension[1] - y2_scaled)

            # Adjust x and y ranges to ensure proper mapping in the array
            x1_final, x2_final = min(x1_scaled, x2_scaled), max(x1_scaled, x2_scaled)
            y1_final, y2_final = min(y1_image, y2_image), max(y1_image, y2_image)

            # Set the obstacle area to 0 in the map
            obstacle_map[y1_final:y2_final, x1_final:x2_final] = 0

        # Convert the map to a JAX array for further processing
        self.obstacle_map = jnp.array(obstacle_map)
        return obstacle_map


    
    def get_value_from_edt_map(self, edt_map, x, y, height):
        array_y = height - 1 - y
        print(array_y, x)

        return edt_map[array_y, x]

    def set_initial_state(self, position, velocity, acceleration):
        self.initial_position = position
        self.initial_velocity = velocity
        self.initial_acceleration = acceleration

    def set_final_state(self, position, velocity, acceleration):
        self.final_position = position
        self.final_velocity = velocity
        self.final_acceleration = acceleration

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    @partial(jit, static_argnums=(0,))
    def compute_minimum_distance_to_obstacles(self, position):
        ########## EXERCISE ##########
        ## Compute the minimum distance to obstacles from current position of the robot.
        ## If you're using the Euclidean distance transformation, define the transformation in the constructor "def __init__".
        
        scale_factor = self.obstacle_map_dim[0] // 10

        # Calculate the continuous position in the array space
        x_continuous = position[0] * scale_factor
        y_continuous = (10 - position[1]) * scale_factor

        # Calculate the discrete indices
        x_index = jnp.floor(x_continuous).astype(jnp.int32)
        y_index = jnp.floor(y_continuous).astype(jnp.int32)

        # Calculate the fractional part to be used for interpolation
        x_frac = x_continuous - x_index
        y_frac = y_continuous - y_index

        # Clip the indices to be within the array bounds
        x_index = jnp.clip(x_index, 0, self.obstacle_map_dim[0] - 2)  # -2 to allow space for interpolation
        y_index = jnp.clip(y_index, 0, self.obstacle_map_dim[1] - 2)

        # Fetch the values for interpolation
        edt_bottom_left = self.edt_map[y_index, x_index]
        edt_bottom_right = self.edt_map[y_index, x_index + 1]
        edt_top_left = self.edt_map[y_index + 1, x_index]
        edt_top_right = self.edt_map[y_index + 1, x_index + 1]

        # Interpolate on the x-axis
        edt_bottom = edt_bottom_left * (1 - x_frac) + edt_bottom_right * x_frac
        edt_top = edt_top_left * (1 - x_frac) + edt_top_right * x_frac

        # Interpolate on the y-axis
        minimum_distance_to_obstacles = edt_bottom * (1 - y_frac) + edt_top * y_frac

            
        #minimum_distance_to_obstacles = 1
        ########## /EXERCISE ##########
        
        return minimum_distance_to_obstacles

    def compute_trajectory(self, x_initial):
        number_of_constraints = 2*6
        
        inf = 1.0e20
        x0 = x_initial

        lb = np.array([ -1]*2*self.number_of_variables,float)
        ub = np.array([ 11]*2*self.number_of_variables,float)

        cl = np.array([0]*number_of_constraints, float)
        cu = np.array([inf]*number_of_constraints, float)
        
        cl[0] = self.initial_position[0]
        cl[1] = self.initial_velocity[0]
        cl[2] = self.initial_acceleration[0]
        cl[3] = self.initial_position[1]
        cl[4] = self.initial_velocity[1]
        cl[5] = self.initial_acceleration[1]
        cl[6] = self.final_position[0]
        cl[7] = self.final_velocity[0]
        cl[8] = self.final_acceleration[0]
        cl[9] = self.final_position[1]
        cl[10] = self.final_velocity[1]
        cl[11] = self.final_acceleration[1]
        
        cu[0] = self.initial_position[0]
        cu[1] = self.initial_velocity[0]
        cu[2] = self.initial_acceleration[0]
        cu[3] = self.initial_position[1]
        cu[4] = self.initial_velocity[1]
        cu[5] = self.initial_acceleration[1]
        cu[6] = self.final_position[0]
        cu[7] = self.final_velocity[0]
        cu[8] = self.final_acceleration[0]
        cu[9] = self.final_position[1]
        cu[10] = self.final_velocity[1]
        cu[11] = self.final_acceleration[1]

        super(BSplineOptimization, self).__init__(n=len(x0),
                                                  m=len(cl),
                                                  lb=lb,
                                                  ub=ub,
                                                  cl=cl,
                                                  cu=cu)

        # self.add_option('print_level', 0)
        # self.add_option('jacobian_approximation', 'finite-difference-values')
        # self.add_option('hessian_approximation', 'limited-memory')
        
        x, info = super(BSplineOptimization, self).solve(x0)

        ## Convert to trajectory data
        control_points_x = x[:self.number_of_variables]
        control_points_y = x[self.number_of_variables:]

        p_x = []
        p_y = []
        for i in range(self.number_of_control_points+1):
            for u in np.arange(0,1,self.dt):
                p_x.append(np.dot(np.dot([u**j for j in range(self.degree_of_bspline)], self.M_3), control_points_x[i:i+self.degree_of_bspline]))
                p_y.append(np.dot(np.dot([u**j for j in range(self.degree_of_bspline)], self.M_3), control_points_y[i:i+self.degree_of_bspline]))
    
        return p_x, p_y, control_points_x, control_points_y
    
    ## ipopt required functions
    @partial(jit, static_argnums=(0,))
    def objective(self, x):
        cost = self.weight_acceleration*jnp.dot(jnp.dot(x, self.MGM_T), x)

        ## Velocity
        for u_M_row in self.u_M_vel:
            v_x = jnp.convolve(u_M_row, x[:self.number_of_variables])
            v_x = v_x[:self.u_M_vel.shape[0]*self.number_of_control_points]
            
            v_y = jnp.convolve(u_M_row, x[self.number_of_variables:])
            v_y = v_y[:self.u_M_vel.shape[0]*self.number_of_control_points]

            cost += self.weight_velocity*jnp.sum(v_x**2+v_y**2)
        
        ## Obstacle avoidance
        for u_M_row in self.u_M_pos:
            pts_x = jnp.convolve(u_M_row, x[:self.number_of_variables])
            pts_x = pts_x[:self.u_M_vel.shape[0]*self.number_of_control_points]
            pts_y = jnp.convolve(u_M_row, x[self.number_of_variables:])
            pts_y = pts_y[:self.u_M_vel.shape[0]*self.number_of_control_points]

            print(pts_x, pts_y)

            minimum_distance_to_obstacles = jnp.array(list(map(self.compute_minimum_distance_to_obstacles, zip(pts_x, pts_y))))

            cost += self.weight_obstacle_avoidance*jnp.sum(1/minimum_distance_to_obstacles)
            
        return cost

    @partial(jit, static_argnums=(0,))
    def constraints(self, x):
        constraints = jnp.array([])

        ## Initial conditions (position, velocity, acceleration)
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([1,0,0]), self.M_3), x[:self.degree_of_bspline])) # x-axis position
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,1,0]), self.M_3), x[:self.degree_of_bspline])) # x-axis velocity
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,0,2]), self.M_3), x[:self.degree_of_bspline])) # x-axis acceleration
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([1,0,0]), self.M_3), x[self.number_of_variables:self.number_of_variables+self.degree_of_bspline])) # y-axis position
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,1,0]), self.M_3), x[self.number_of_variables:self.number_of_variables+self.degree_of_bspline])) # y-axis velocity
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,0,2]), self.M_3), x[self.number_of_variables:self.number_of_variables+self.degree_of_bspline])) # y-axis acceleration
        
        ## Final conditions (position, velocity, acceleration)
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([1,1,1]), self.M_3), x[self.number_of_variables-self.degree_of_bspline:self.number_of_variables])) # x-axis position
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,1,2]), self.M_3), x[self.number_of_variables-self.degree_of_bspline:self.number_of_variables])) # x-axis velocity
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,0,2]), self.M_3), x[self.number_of_variables-self.degree_of_bspline:self.number_of_variables])) # x-axis acceleration
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([1,1,1]), self.M_3), x[2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables])) # y-axis position
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,1,2]), self.M_3), x[2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables])) # y-axis velocity
        constraints = jnp.append(constraints, jnp.dot(jnp.dot(jnp.array([0,0,2]), self.M_3), x[2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables])) # y-axis acceleration

        return jnp.array(constraints)

    def jacobianstructure(self):
        number_of_constraints = 2*6 + \
            self.number_of_control_points*int(np.ceil(1/self.dt))
        jacobian_structure = np.zeros((number_of_constraints, 2*self.number_of_variables), int)

        ## Initial conditions (position, velocity, acceleration)
        jacobian_structure[0,:self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[1,:self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[2,:self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[3, self.number_of_variables:self.number_of_variables+self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[4, self.number_of_variables:self.number_of_variables+self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[5, self.number_of_variables:self.number_of_variables+self.degree_of_bspline] = np.ones(self.degree_of_bspline, int)

        ## Final conditions (position, velocity, acceleration)
        jacobian_structure[6, self.number_of_variables-self.degree_of_bspline:self.number_of_variables] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[7, self.number_of_variables-self.degree_of_bspline:self.number_of_variables] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[8, self.number_of_variables-self.degree_of_bspline:self.number_of_variables] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[9, 2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[10, 2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables] = np.ones(self.degree_of_bspline, int)
        jacobian_structure[11, 2*self.number_of_variables-self.degree_of_bspline:2*self.number_of_variables] = np.ones(self.degree_of_bspline, int)

        jacobian_structure = sps.coo_matrix(jacobian_structure)

        return (jnp.array(jacobian_structure.row), jnp.array(jacobian_structure.col))

    @partial(jit, static_argnums=(0,))
    def jacobian(self, x):
        jacobian_matrix = jacfwd(self.constraints)(x)
        row, col = self.jacobianstructure()

        return jacobian_matrix[row, col]

    @partial(jit, static_argnums=(0,))
    def cost_hessian_reduced(self, x):
        row, col = self.hessianstructure()
        hessian_matrix_reduced = jacrev(jacfwd(self.objective))(x)[row, col]

        return hessian_matrix_reduced

    @partial(jit, static_argnums=(0,))
    def constraints_hessian_reduced(self, x):
        row, col = self.hessianstructure()
        hessian_matrix_reduced = jacfwd(jacrev(self.constraints))(x)[:,row,col]

        return hessian_matrix_reduced
    
    def hessianstructure(self):
        hessian_structure = np.tril(np.zeros((2*self.number_of_variables, 2*self.number_of_variables), int))

        for i in range(2*self.number_of_variables):
            hessian_structure[i,i] = 1

            for j in range(1,self.degree_of_bspline):
                if (i < 2*self.number_of_variables-j):
                    hessian_structure[i+j,i] = 1

        hessian_structure = sps.coo_matrix(hessian_structure)
        
        return (jnp.array(hessian_structure.row), jnp.array(hessian_structure.col))

    @partial(jit, static_argnums=(0,))
    def hessian(self, x, lagrange, obj_factor):
        hessian_cost = self.cost_hessian_reduced(x)
        hessian_constraints = self.constraints_hessian_reduced(x)

        hessian = obj_factor*hessian_cost

        return hessian
