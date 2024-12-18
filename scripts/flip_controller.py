import sys
import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import ast
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ur_kinematics import *
from research_main.envs.mujoco_utils import *
from research_main.envs.sim_utils import *
from scipy.spatial.transform import Rotation as R



class Controller:
    def __init__(self, model, data, use_mode):
        self.state = 'initial_pose'
        self.move_to_next_state = True
        self.has_block_released = False
        self.use_mode = use_mode
        

        self.joint_ids = {
            'shoulder_pan_joint': model.joint('shoulder_pan_joint').id,
            'shoulder_lift_joint': model.joint('shoulder_lift_joint').id,
            'elbow_joint': model.joint('elbow_joint').id,
            'wrist_1_joint': model.joint('wrist_1_joint').id,
            'wrist_2_joint': model.joint('wrist_2_joint').id,
            'wrist_3_joint': model.joint('wrist_3_joint').id
        }


        if self.use_mode == "RL_eval":
            self.active_motors_list = [
                self.joint_ids['shoulder_lift_joint'], 
                self.joint_ids['elbow_joint'], 
                self.joint_ids['wrist_1_joint'],
            ]
            self.passive_motors_list = [
                self.joint_ids['shoulder_pan_joint'],
                self.joint_ids['wrist_2_joint'], 
                self.joint_ids['wrist_3_joint']
            ]

            self.active_motors = ActuatorController(self.active_motors_list)
            self.passive_motors = ActuatorController(self.passive_motors_list)
            self.active_motors.switch_to_velocity_controller(model)
            self.passive_motors.switch_to_position_controller(model)

            self.fixed_qpos_values = data.qpos[self.passive_motors_list].copy()
            data.ctrl[self.passive_motors_list] = self.fixed_qpos_values

            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(os.path.join(current_dir, "..", "research_main", "envs"))
            config_path = os.path.join(base_dir, "release_state_config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
        
            release_state_config = config["release_state"]
            self.desired_release_state = {
                "v_x0": release_state_config["v_x0"],
                "v_y0": release_state_config["v_y0"],
                "v_z0": release_state_config["v_z0"],
                "theta_x0": release_state_config["theta_x0"],
                "theta_y0": np.pi - release_state_config["theta_y0"],  
                "theta_z0": release_state_config["theta_z0"],
                "omega_x0": release_state_config["omega_x0"],
                "omega_y0": release_state_config["omega_y0"],
                "omega_z0": release_state_config["omega_z0"],
                "h_0": release_state_config["h_0"]
            }


        else:
            self.active_motors_list = [
                self.joint_ids['shoulder_pan_joint'],
                self.joint_ids['shoulder_lift_joint'], 
                self.joint_ids['elbow_joint'], 
                self.joint_ids['wrist_1_joint'],
                self.joint_ids['wrist_2_joint'], 
                self.joint_ids['wrist_3_joint']
            ]
            self.passive_motors_list = [
            ]

            self.active_motors = ActuatorController(self.active_motors_list)
            self.passive_motors = ActuatorController(self.passive_motors_list)
            self.active_motors.switch_to_velocity_controller(model)
            self.passive_motors.switch_to_velocity_controller(model)

            self.fixed_qpos_values = data.qpos[self.active_motors_list].copy()
            data.ctrl[self.active_motors_list] = self.fixed_qpos_values
        
        gripper_close(data)
        
        
        self.has_gripper_opened = False
        self.passive_motor_angles_hold = None

        self.trigger_iteration = 0
        self.current_ee_pose = 0
        self.traj_timestep = 0

        self.time_hist = []
        self.joint_vel_hist = []
        self.target_joint_vel_hist = []
        self.ee_linvel_hist = []
        self.ee_angvel_hist = []

        self.simulation_stop = False


    def flip_block(self, model, data, time, ee_flip_target_velocity):
        self.time_hist.append(time)
        
        joint_velocities = np.array(get_joint_velocities(data)).flatten()
        self.joint_vel_hist.append(joint_velocities)
        
        
        self.current_ee_pose = np.copy(get_ee_pose(model, data)[0])
        quat_current_ee = np.array(data.sensor('pinch_quat').data.copy())
        ee_linvel = data.sensor('pinch_linvel').data.copy()
        ee_angvel = data.sensor('pinch_angvel').data.copy()


        if self.has_gripper_opened == False:
            target_trans_velocities = ee_flip_target_velocity
            target_ang_velocities = [0, -np.pi, 0]
            target_joint_velocities = calculate_joint_velocities_trans_partial(model, data, target_trans_velocities, target_ang_velocities)
            self.target_joint_vel_hist.append(target_joint_velocities.copy())

            _, block_orientation = get_block_pose(model, data, 'block_0')

            shoulder_lift_vel_bias = -0.125
            elbow_vel_bias = -0.075
            biased_target_joint_velocities = target_joint_velocities.copy()
            biased_target_joint_velocities[1] += shoulder_lift_vel_bias
            biased_target_joint_velocities[2] += elbow_vel_bias
            biased_target_joint_velocities[3] = -5*np.pi/6
            set_joint_states(data, self.active_motors_list, biased_target_joint_velocities)

            
            if block_orientation[1] < -1.042:
                gripper_open(data)
                self.has_gripper_opened = True
                self.has_block_flipped = True
                self.release_time = time
                self.release_ee_height = self.current_ee_pose[2]
                self.release_ee_quat = quat_current_ee
                self.release_ee_linvel = ee_linvel
                self.release_ee_angvel = ee_angvel

                self.passive_motor_angles_hold = get_specific_joint_angles(data, self.passive_motors_list)       
    

        else:
            self.target_joint_vel_hist.append(np.zeros((6)))
            self.trigger_iteration += 1
            set_joint_states(data, self.active_motors_list, np.zeros(6))

  

    def execute_flip_trajectory(self, model, data, time, traj_ctrl, frameskip, policy_version, policy_type):
        quat_desired_block = R.from_euler('xyz', [0, np.pi-2.0945, -3.14]).as_quat()
        quat_current_block = np.array(data.sensor('block_quat').data.copy()) 

        quat_current_ee = np.array(data.sensor('pinch_quat').data.copy())
        quat_dist = quat_distance(quat_current_block, quat_desired_block)

        joint_velocities = np.array(get_joint_velocities(data)).flatten()
        ee_pos = data.sensor('pinch_pos').data.copy()
        ee_linvel = data.sensor('pinch_linvel').data.copy()
        ee_angvel = data.sensor('pinch_angvel').data.copy()
        self.joint_vel_hist.append(joint_velocities)
        self.time_hist.append(time)
        self.ee_linvel_hist.append(ee_linvel)
        self.ee_angvel_hist.append(ee_angvel)

        #if self.mpc_timestep < len(self.RL_ctrl) and quat_dist > 5e-3:
        if self.traj_timestep < len(traj_ctrl):
            given_ctrl = traj_ctrl[self.traj_timestep]
            data.ctrl[:6] = given_ctrl
        

    #print(self.mpc_timestep, len(self.RL_ctrl))
        else:
            gripper_open(data)
            if self.trigger_iteration == 0:
                self.active_motors.switch_to_position_controller(model)                
                self.hold_qpos_values = data.qpos[:6].copy()
                self.has_gripper_opened = True
                self.release_time = time
                self.release_ee_height = ee_pos[2]
                self.release_ee_quat = quat_current_ee
                self.release_ee_linvel = ee_linvel
                self.release_ee_angvel = ee_angvel
                self._log_mujoco_release_state(policy_version, policy_type)
                self._compare_release_states(policy_version, policy_type)
                self.trigger_iteration += 1
            data.ctrl[:6] = self.hold_qpos_values

        # Log data at each time step
        if not hasattr(self, 'csv_initialized'):
            with open('execute_trajectory_log.csv', 'w', newline='') as csvfile:
                log_writer = csv.writer(csvfile)
                log_writer.writerow(['Time', 'Ctrl', 'QPos', 'QVel'])  # Write headers
            self.csv_initialized = True  

        with open(f'../trajectories/RL_trajectory/{policy_version}/executed_{policy_type}trajectory.csv', 'a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            ctrl_str = ','.join(map(str, data.ctrl[:6].copy()))
            qpos_str = ','.join(map(str, data.qpos[:6].copy()))
            qvel_str = ','.join(map(str, data.qvel[:6].copy()))
            log_writer.writerow([time, ctrl_str, qpos_str, qvel_str])

        # print(self.traj_timestep, len(traj_ctrl))
        self.traj_timestep += frameskip

    def _log_mujoco_release_state(self, policy_version, policy_type):
        release_data = {
            "ee_linvel": self.release_ee_linvel.tolist(),
            "ee_angvel": self.release_ee_angvel.tolist(),
            "ee_height": self.release_ee_height,
            "ee_quat": self.release_ee_quat.tolist(),
        }

        with open(f"../trajectories/RL_trajectory/{policy_version}/executed_{policy_type}_release_states.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["EE LinVel", "EE AngVel", "EE Height", "EE Quat"])
            writer.writerow([
                release_data["ee_linvel"], 
                release_data["ee_angvel"], 
                release_data["ee_height"], 
                release_data["ee_quat"]
            ])

        #print(f"MuJoCo Release State: {release_data}")



    def _compare_release_states(self, policy_version, policy_type):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        base_dir = os.path.abspath(os.path.join(current_dir, "..", "trajectories", "RL_trajectory"))
        policy_dir = os.path.join(base_dir, policy_version)
        
        rl_file = os.path.join(policy_dir, f"raw_{policy_type}_release_states.csv")
        mujoco_file = os.path.join(policy_dir, f"executed_{policy_type}_release_states.csv")

        rl_release = self._load_release_state(rl_file)
        mujoco_release = self._load_release_state(mujoco_file)
        desired_state = self.desired_release_state

        desired_quat = R.from_euler('xyz', [
            desired_state['theta_x0'], 
            desired_state['theta_y0'], 
            desired_state['theta_z0']
        ]).as_quat()
        desired_euler = R.from_quat(desired_quat).as_euler('xyz', degrees=True)

        rl_euler = R.from_quat(rl_release['quat']).as_euler('xyz', degrees=True)
        mujoco_euler = R.from_quat(mujoco_release['quat']).as_euler('xyz', degrees=True)

        results = {
            "RL_vs_MuJoCo": {
                "RL_linear_velocity": rl_release['linvel'],
                "MuJoCo_linear_velocity": mujoco_release['linvel'],
                "linvel_difference": np.linalg.norm(np.array(rl_release['linvel']) - np.array(mujoco_release['linvel'])),
                "RL_angular_velocity": rl_release['angvel'],
                "MuJoCo_angular_velocity": mujoco_release['angvel'],
                "angvel_difference": np.linalg.norm(np.array(rl_release['angvel']) - np.array(mujoco_release['angvel'])),
                "RL_height": rl_release['height'],
                "MuJoCo_height": mujoco_release['height'],
                "height_difference": abs(rl_release['height'] - mujoco_release['height']),
                "RL_quat": rl_release['quat'],
                "RL_euler_xyz": rl_euler.tolist(),
                "MuJoCo_quat": mujoco_release['quat'],
                "MuJoCo_euler_xyz": mujoco_euler.tolist(),
                "quat_difference (norm)": np.linalg.norm(np.array(rl_release['quat']) - np.array(mujoco_release['quat']))
            },
            "MuJoCo_vs_Desired": {
                "MuJoCo_linear_velocity": mujoco_release['linvel'],
                "Desired_linear_velocity": [desired_state['v_x0'], desired_state['v_y0'], desired_state['v_z0']],
                "linvel_difference": np.linalg.norm(np.array(mujoco_release['linvel']) - np.array([desired_state['v_x0'], desired_state['v_y0'], desired_state['v_z0']])),
                "MuJoCo_angular_velocity": mujoco_release['angvel'],
                "Desired_angular_velocity": [desired_state['omega_x0'], desired_state['omega_y0'], desired_state['omega_z0']],
                "angvel_difference": np.linalg.norm(np.array(mujoco_release['angvel']) - np.array([desired_state['omega_x0'], desired_state['omega_y0'], desired_state['omega_z0']])),
                "MuJoCo_height": mujoco_release['height'],
                "Desired_height": desired_state['h_0'],
                "height_difference": abs(mujoco_release['height'] - desired_state['h_0']),
                "MuJoCo_quat": mujoco_release['quat'],
                "MuJoCo_euler_xyz": mujoco_euler.tolist(),
                "Desired_quat": desired_quat.tolist(),
                "Desired_euler_xyz": desired_euler.tolist(),
                "quat_difference (dist)": quat_distance_new(mujoco_release['quat'], desired_quat)
            }
        }

        result_file_path = os.path.join(policy_dir, f"{policy_type}_comparison_results.json")
        with open(result_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # Print results for both comparisons
        print("=" * 91)
        print("SECTION 1: RL Evaluation vs MuJoCo Execution")
        print(json.dumps(results["RL_vs_MuJoCo"], indent=4))
        print("=" * 91)
        print("\nSECTION 2: MuJoCo Execution vs Desired Release State")
        print(json.dumps(results["MuJoCo_vs_Desired"], indent=4))
        print("=" * 91)
        print(f"Comparison results saved to: {result_file_path}")


    def _load_release_state(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  
            data = next(reader) 
            return {
                "linvel": ast.literal_eval(data[0]),  
                "angvel": ast.literal_eval(data[1]),
                "height": float(data[2]),
                "quat": ast.literal_eval(data[3])  
            }
