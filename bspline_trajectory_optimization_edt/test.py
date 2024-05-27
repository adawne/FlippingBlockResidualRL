from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

trajectory_file = 'grasp_pick_trajectory_box_4.pkl'

def load_data_with_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)  # Assuming data is a dictionary
        return data
    except Exception as e:
        print(f"Failed to load or parse the file: {e}")
        return None

data = load_data_with_pickle(trajectory_file)

p_x = data['p_x']
p_y = data['p_y']
p_x = np.array(p_x)
p_y = np.array(p_y)

p_x = p_x - 2.7
p_y = p_y - 2.7

# Scaling
p_x = p_x * 0.2
p_y = p_y * 0.2

print("Length of p_x: ", len(p_x))

t = np.linspace(0, 1, len(p_x))
interp_func_x = interp1d(t, p_x, kind='cubic')
interp_func_y = interp1d(t, p_y, kind='cubic')
t_new = np.linspace(0, 1, 1000)  

dense_pts_x = interp_func_x(t_new)
dense_pts_y = interp_func_y(t_new)

p_x = dense_pts_x
p_y = dense_pts_y
print("Last point: ", p_x[-1], p_y[-1])
print("First point: ", p_x[0], p_y[0])

print("Length of p_x: ", len(p_x))

# Assuming the relationship is with x, and p_x is sorted
z_start = 0.05
z_end = 0.03
#z = np.interp(p_x, [min(p_x), max(p_x)], [z_end, z_start]) 
p_z = 0.05 * np.ones(len(p_x))

# Visualization in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(p_x, p_y, p_z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
