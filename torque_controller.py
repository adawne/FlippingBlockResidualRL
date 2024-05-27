import numpy as np
import pybullet as pb
import control

from kuka_dynamics import *
from kuka_dynamics import iterative_ne_algorithm

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]


def compute_control_gain(mode):
    n = 7  # Number of joints
    m = 14  # State space size (joint angles and velocities)

    A = np.zeros((m, m))
    A[7:m, 0:7] = np.eye(7)
    B = np.zeros((m, n))
    B[0:n, 0:n] = np.eye(n)  # Adjusted to affect the second half of A

    Q = np.eye(m)
    for i in range(7, 14):
        Q[i, i] = 1000

    if mode == 'accurate':
        R = 0.0001 * np.eye(n)
        for i in range(7, 14):
            Q[i, i] = 10000
    elif mode == 'fast':
        R = 0.001 * np.eye(n)
    elif mode =='slow':
        R = 0.1*np.eye(n)
    else:
        R = np.eye(n)
        for i in range(7, 14):
            Q[i, i] = 1

    K, S, E = control.lqr(A, B, Q, R)
    K_1 = -K[0:n, 0:n]
    K_2 = -K[0:n, 7:m]
    
    return K_1, K_2


def compute_joint_torques_move_to_point(K_1, K_2, joint_angles, joint_angles_target, joint_velocities, joint_velocities_target, u_bar_ref ):

    tau_bar = u_bar_ref + K_1@(joint_velocities-joint_velocities_target) + K_2@(joint_angles - joint_angles_target)
    tau = iterative_ne_algorithm(len(arm_joint_indices), joint_angles, joint_velocities, tau_bar)

    return tau


