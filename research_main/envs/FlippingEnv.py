import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils.env_checker import check_env

from research_main.envs.mujoco_utils import *
from research_main.envs.scene_builder import *

# from mujoco_utils import *
# from scene_builder import *


DEFAULT_CAMERA_CONFIG = {
    "distance" : 3,
    "azimuth" : 130,
    "elevation" : -25
}

class URFlipBlockEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, render_mode=None, use_mpc=False, use_qpos=True):
        block_position_orientation = [([0.2, 0.2, 0], [0, 0, 0])]
        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_position_orientation, use_mode="RL_train")

        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(world_xml_model)
        self.data = mujoco.MjData(self.model)

        self.use_mpc = use_mpc
        self.use_qpos = use_qpos

        self.joint_ids = {
            'shoulder_pan_joint': self.model.joint('shoulder_pan_joint').id,
            'shoulder_lift_joint': self.model.joint('shoulder_lift_joint').id,
            'elbow_joint': self.model.joint('elbow_joint').id,
            'wrist_1_joint': self.model.joint('wrist_1_joint').id,
            'wrist_2_joint': self.model.joint('wrist_2_joint').id,
            'wrist_3_joint': self.model.joint('wrist_3_joint').id
        }

        self.active_motors_list = [
            self.joint_ids['shoulder_lift_joint'], 
            self.joint_ids['elbow_joint'], 
            self.joint_ids['wrist_1_joint']
        ]
        self.passive_motors_list = [
            self.joint_ids['shoulder_pan_joint'], 
            self.joint_ids['wrist_2_joint'], 
            self.joint_ids['wrist_3_joint']
        ]

        self.active_motors = ActuatorController(self.active_motors_list)
        

        self.passive_motors = ActuatorController(self.passive_motors_list)
        if self.use_qpos:
            self.active_motors.switch_to_position_controller(self.model)
        else:
            self.active_motors.switch_to_velocity_controller(self.model)



        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.fixed_qpos_values = self.data.qpos[self.passive_motors_list].copy()
        gripper_close(self.data)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.model.opt.timestep)),
        }

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config=DEFAULT_CAMERA_CONFIG,
        )

        if self.use_qpos:
            self.qpos_min = np.array([-2 * np.pi, -np.pi, -2 * np.pi])
            self.qpos_max = np.array([2 * np.pi, np.pi, 2 * np.pi])
        else: 
            self.qvel_min = -np.array([120, 180, 180]) * np.pi / 180
            self.qvel_max = np.array([120, 180, 180]) * np.pi / 180

        if self.use_mpc:
            self.mpc_nominal_traj = []
            with open('research_main/envs/precomputed_mpc_traj/interpolated_trajectory.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if self.use_qpos:
                        ctrl_values = [float(x) for x in row['QPos'].split(',')]
                    else:
                        ctrl_values = [float(x) for x in row['QVel'].split(',')]
                    self.mpc_nominal_traj.append([ctrl_values[i] for i in [1, 2, 3]])

            self.mpc_nominal_traj = np.array(self.mpc_nominal_traj)

            if self.use_qpos:
                self.residual_low = np.max(self.qpos_min - self.mpc_nominal_traj, axis=0)
                self.residual_high = np.min(self.qpos_max - self.mpc_nominal_traj, axis=0)
            else:
                self.residual_low = np.max(self.qvel_min - self.mpc_nominal_traj, axis=0)
                self.residual_high = np.min(self.qvel_max - self.mpc_nominal_traj, axis=0)
            self.mpc_timestep = 0
        #print(f"Action Residual Bound: {self.residual_low} - {self.residual_high}")
 
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len([1, 2, 3]),),  
            dtype=np.float32
        )

        self.valid_flip = True
        self.trigger_iteration = 0
        self.initial_block_quat = [1, 0, 0, 0]
        self.quat_release_desired_block = R.from_euler('xyz', [0, np.pi-2.0945, -3.14]).as_quat()
        

        # Desired release state
        self.desired_release_state = {
            'v_x0': 0.2,
            'v_y0': 0,
            'v_z0': 0.5864,
            'theta_x0': 0,
            'theta_y0': np.pi - 2.0945,
            'theta_z0': -3.14,
            'omega_x0': 0,
            'omega_y0': -0.314,
            'omega_z0': 0,
            'h_0': 0.35
        }

        self.block_linvel = self.data.sensor('block_linvel').data.copy()
        self.block_angvel = self.data.sensor('block_angvel').data.copy()
        self.block_quat = self.data.sensor('block_quat').data.copy()
        self.block_position, _ = get_block_pose(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        gripper_close(self.data)

        self.valid_flip = True
        self.trigger_iteration = 0

        if self.use_mpc:
            self.mpc_timestep = 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _normalize_action(self, action):
        action = np.array(action)  

        if self.use_mpc:
            return self.residual_low + 0.5 * (action + 1) * (self.residual_high - self.residual_low)
        else:
            if self.use_qpos:
                return self.qpos_min + 0.5 * (action + 1) * (self.qpos_max - self.qpos_min)
            else:
                return self.qvel_min + 0.5 * (action + 1) * (self.qvel_max - self.qvel_min)


    def step(self, action):
        reward = 0
        terminated = False
        frame_skip = 5

        self.block_linvel = self.data.sensor('block_linvel').data.copy()
        self.block_angvel = self.data.sensor('block_angvel').data.copy()
        self.block_quat = self.data.sensor('block_quat').data.copy()
        self.block_position, _ = get_block_pose(self.model, self.data)
        
        if self.use_mpc:
            mpc_action = self.mpc_nominal_traj[self.mpc_timestep]
        normalized_action = self._normalize_action(action)

        if self.use_mpc:
            final_action = mpc_action + normalized_action
            #final_action = mpc_action 
            self.mpc_timestep += frame_skip
            # print(f"Action original: {action} | Normalized action: {normalized_action} | MPC action: {mpc_action} | Final action: {final_action} {self.data.ctrl[:6].copy()}")

        else:
            final_action = normalized_action
            # print(f"Action original: {action} | Normalized action: {normalized_action} | Final action: {final_action} {self.data.ctrl[:6].copy()}")

        self.data.ctrl[self.passive_motors_list] = self.fixed_qpos_values
        self.data.ctrl[self.active_motors_list] = final_action

        mujoco.mj_step(self.model, self.data, nstep=frame_skip)

        reward, terminated = self._compute_reward()

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info


    def _compute_reward(self):
        block_height = self.block_position[2]
        quat_dist = quat_distance(self.block_quat, self.quat_release_desired_block)
        active_qvel = self.data.qvel[self.active_motors_list].copy() 
        qvel_limits = np.array([120 * np.pi / 180, 180 * np.pi / 180, 180 * np.pi / 180])

        if has_block_hit_floor(self.data):
            # print("Hit floor before flip")
            return -20, True

        if self.use_qpos:
            if np.any(active_qvel < -qvel_limits) or np.any(active_qvel > qvel_limits):
                # print("Joint velocity limit exceeded")
                return -20, True

        if self._is_close_to_desired_state():
            # print("Close to desired state")
            return 20, True

        position_error = abs(block_height - self.desired_release_state['h_0'])
        translational_velocity_error = np.linalg.norm(
            self.block_linvel - [self.desired_release_state['v_x0'], self.desired_release_state['v_y0'], self.desired_release_state['v_z0']]
        )
        angular_velocity_error = np.linalg.norm(
            self.block_angvel - [self.desired_release_state['omega_x0'], self.desired_release_state['omega_y0'], self.desired_release_state['omega_z0']]
        )
        desired_quat = R.from_euler(
            'xyz', [self.desired_release_state['theta_x0'], self.desired_release_state['theta_y0'], self.desired_release_state['theta_z0']]
        ).as_quat()
        orientation_error = quat_distance_new(self.block_quat, desired_quat)

        progress_reward = max(0, 1 - (position_error + translational_velocity_error))
        reward = (
            0.5 * progress_reward
            - 0.1 * position_error
            - 0.1 * translational_velocity_error
            - 0.1 * angular_velocity_error
            - 0.1 * orientation_error
        )
        # print("Discrepancy between desired state and current state", progress_reward, reward)

        if self.use_mpc and self.mpc_timestep >= len(self.mpc_nominal_traj):
            # print("MPC Timestep or angle pass")
            return reward, True 
        
        if quat_dist < 5e-3:
            # print("Angle Pass")
            return reward, True

        return reward, False
     
    def _get_obs(self): 
        joint_pos = self.data.qpos[self.active_motors_list].copy() 
        joint_vel = self.data.qvel[self.active_motors_list].copy()  
        ee_trans_vel = self.data.sensor('pinch_linvel').data.copy()  
        ee_ang_vel = self.data.sensor('pinch_angvel').data.copy()    
        ee_quat = self.data.sensor('pinch_quat').data.copy()         
        ee_height = self.data.sensor('pinch_pos').data.copy()[2]     
        
        obs = np.concatenate([
            joint_pos,         
            joint_vel,         
            [ee_height],       
            ee_quat,           
            ee_trans_vel,      
            ee_ang_vel,        
        ])
        return obs

    def _calculate_residuals(self):
        translational_velocity_residual = np.linalg.norm(
            self.block_linvel - np.array([self.desired_release_state['v_x0'], 
                                        self.desired_release_state['v_y0'], 
                                        self.desired_release_state['v_z0']])
        )
        angular_velocity_residual = np.linalg.norm(
            self.block_angvel - np.array([self.desired_release_state['omega_x0'], 
                                        self.desired_release_state['omega_y0'], 
                                        self.desired_release_state['omega_z0']])
        )
        height_residual = np.abs(self.block_position[2] - self.desired_release_state['h_0'])
        desired_release_quat = R.from_euler(
            'xyz', 
            [self.desired_release_state['theta_x0'], 
            self.desired_release_state['theta_y0'], 
            self.desired_release_state['theta_z0']]
        ).as_quat()
        orientation_residual = quat_distance_new(self.block_quat, desired_release_quat)

        return translational_velocity_residual, angular_velocity_residual, height_residual, orientation_residual

    def _is_close_to_desired_state(self):
        translational_velocity_residual, angular_velocity_residual, height_residual, orientation_residual = self._calculate_residuals()

        return (
            translational_velocity_residual < 0.05 and
            angular_velocity_residual < 0.05 and
            height_residual < 0.05 and
            orientation_residual < 0.01
        )

    def _get_info(self):
        obs = self._get_obs()

        joint_pos = obs[:3]
        joint_vel = obs[3:6]
        translational_velocity_residual, angular_velocity_residual, height_residual, orientation_residual = self._calculate_residuals()

        info = {
            "joint_positions": joint_pos.tolist(),
            "joint_velocities": joint_vel.tolist(),
            "ee_height_residual": height_residual,
            "translational_velocity_residual": translational_velocity_residual,
            "angular_velocity_residual": angular_velocity_residual,
            "orientation_residual": orientation_residual,
            "valid_flip": self.valid_flip,
        }

        if self.use_mpc:  # Add mpc_timestep only if self.use_mpc is True
            info["mpc_timestep"] = self.mpc_timestep

        return info


    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        
    def seed(self, seed):
        np.random.seed(seed)

# def manual_test():
#     env = URFlipBlockEnv(render_mode='human')  # Initialize the environment
#     num_episodes = 1  # Number of episodes to test
#     print(f"Starting manual test with {num_episodes} episodes...")
#     print(f"Motor list: {env.active_motors_list}")

#     for episode in range(num_episodes):
#         observation, info = env.reset()
#         print(f"--- Episode {episode + 1}/{num_episodes} ---")
#         print("Initial observation:", observation)
#         print("Action space (low):", env.action_space.low)
#         print("Action space (high):", env.action_space.high)
#         print(f"Low normalized_action = {env._normalize_action(env.action_space.low)}")
#         print(f"High normalized_action = {env._normalize_action(env.action_space.high)}")

#         total_reward = 0
#         terminated = False
#         action_count = 0
#         frame_skip = 5

#         while not terminated:
#             print("="*25)
#             print(f"Step {action_count*frame_skip}:")
#             #action = env.action_space.sample()  # Generate a random action
#             action = [-1, -1, -1]
#             observation, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
#             action_count += 1

#             # # Print step information
#             print(f"Action: {action}")
#             print(f"Observation: {observation}")
#             print(f"Joint pos: {info['joint_positions']}")
#             print(f"Joint vel: {info['joint_velocities']}")
#             print(f"Reward: {reward}")
#             print(f"Terminated: {terminated}")

#         print(f"--- Episode {episode + 1} Summary ---")
#         print(f"Total reward: {total_reward}")
#         print(f"Total steps: {action_count}")
#         print(f"Block pose: {get_block_pose(env.model, env.data, 'block_0')}")

#     env.close()
#     print("Manual test completed.")


# if __name__ == "__main__":
#     manual_test()

#     # env = URFlipBlockEnv(render_mode='human')
#     # check_env(env, skip_render_check=False)

