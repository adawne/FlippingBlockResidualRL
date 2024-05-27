import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import cyipopt
import pickle

from shapely.geometry import Polygon

from BSplineOptimizationJAXEDT import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    ## Trajectory planning parameters
    origin = (5.8, 3.3)
    destination = (2.75, 0)
    obstacles = [] # list of obstacles' polygonal representation (points)
    #obstacles.append(Polygon([[3,2],
    #                          [4,2],
    #                          [4,4],
    #                          [3,4],
    #                          [3,2]]))




    bspline = BSplineOptimization(number_of_control_points=10, dt=.1)
    bspline.set_initial_state(position=origin, velocity=(0,0), acceleration=(0,0))
    bspline.set_final_state(position=destination, velocity=(0,0), acceleration=(0,0))
    bspline.set_obstacles(obstacles)

    x_initial = 1*np.random.random(2*bspline.number_of_variables)
    p_x, p_y, control_points_x, control_points_y = bspline.compute_trajectory(x_initial)

    print("Optimization result: ", p_x, p_y)
    print("Length of p_x: ", len(p_x), len(p_y))
    optimization_result = {
        'p_x': p_x,
        'p_y': p_y,
        'control_points_x': control_points_x,
        'control_points_y': control_points_y
    }
    with open('grasp_pick_trajectory_box_3.pkl', 'wb') as f:
        pickle.dump(optimization_result, f)
    
    ## Plot the environment
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        plt.fill(*obstacle.exterior.xy, color='black')

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()

    ## Plot results
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ## Plot control points
    for i in range(bspline.number_of_variables):
        ax.plot(control_points_x[i], control_points_y[i], 'k.')
        
    ## Plot trajectory
    print(p_x, p_y)
    ax.plot(p_x, p_y)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        plt.fill(*obstacle.exterior.xy, color='black')

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()
        

