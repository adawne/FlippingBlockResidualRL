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

    def __init__(self, render_mode=None):
        block_position_orientation = [([0.2, 0.2, 0], [0, 0, 0])]
        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_position_orientation, use_mode = "RL_train")
        
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(world_xml_model)
        self.data = mujoco.MjData(self.model)
        joint_ids = [self.model.joint(name).id for name in [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]]

        self.active_motors_list = joint_ids
        self.active_motors = ActuatorController(self.active_motors_list)
        self.active_motors.switch_to_velocity_controller(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
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
            default_cam_config = DEFAULT_CAMERA_CONFIG,
            # width=1440,
            # height=1024,
        )

        qvel_min = np.array([-2.09, -2.09, -3.14, -3.14, -3.14, -3.14])
        qvel_max = np.array([2.09, 2.09, 3.14, 3.14, 3.14, 3.14])

        self.mpc_nominal_traj = []
        with open('research_main/envs/precomputed_mpc_traj/interpolated_trajectory.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ctrl_values = [float(x) for x in row['QVel'].split(',')]  
                self.mpc_nominal_traj.append(ctrl_values[:6])  

        self.mpc_nominal_traj = np.array(self.mpc_nominal_traj)
        residual_low_t = qvel_min - self.mpc_nominal_traj
        residual_high_t = qvel_max - self.mpc_nominal_traj

        self.mpc_nominal_traj = np.array(self.mpc_nominal_traj)
        self.residual_low = np.max(qvel_min - self.mpc_nominal_traj, axis=0) 
        self.residual_high = np.min(qvel_max - self.mpc_nominal_traj, axis=0)

        # Manipulator: joint pos(6), joint vel(6) | Manipulator: EE [quat[4], transvel[3], angvel[3], h]  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.mpc_nominal_traj.shape[1],),  
            dtype=np.float32
        )
        self.has_gripper_opened = False
        self.has_block_released = False
        self.has_block_landed = False
        self.has_block_steady = False
        self.valid_flip = True
        self.trigger_iteration = 0
        self.mpc_timestep = 0
        self.initial_block_quat = [1, 0, 0, 0]
        # TODO: Fetch the result from the optimizer directly
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

        self.has_gripper_opened = False
        self.has_block_released = False
        self.has_block_landed = False
        self.has_block_steady = False
        self.valid_flip = True
        self.trigger_iteration = 0
        self.mpc_timestep = 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _normalize_action(self, action):
            return self.residual_low + 0.5 * (action + 1) * (self.residual_high - self.residual_low)


    def step(self, action):
        reward = 0
        terminated = False

        self.block_linvel = self.data.sensor('block_linvel').data.copy()
        self.block_angvel = self.data.sensor('block_angvel').data.copy()
        self.block_quat = self.data.sensor('block_quat').data.copy()
        self.block_position, _ = get_block_pose(self.model, self.data)

        if not self.has_gripper_opened:
            mpc_action = self.mpc_nominal_traj[self.mpc_timestep]
            normalized_action = self._normalize_action(action)
            final_action = mpc_action + normalized_action
            #final_action = normalized_action

            self.data.ctrl[:6] = final_action

            #print(f"Action original: {action} | Normalized action: {normalized_action} | MPC action: {mpc_action} | Final action: {final_action} {self.data.ctrl[:6].copy()}")
            #print(f"Action: {normalized_action} | {self.data.ctrl.copy()}")

            mujoco.mj_step(self.model, self.data)
            self.mpc_timestep += 1

            if self.mpc_timestep >= len(self.mpc_nominal_traj) or self._is_close_to_desired_state():
                #print("Gripper open triggered")
                gripper_open(self.data)
                self.has_gripper_opened = True

        if self.has_gripper_opened:
            self._wait_for_block_to_land()

        # FIXME: Logic kalo gripper dah kebuka dan balok kena robot harus diapain keknya kondisional valid_flip gaperlu, diganti aja sama block touch robto
        reward, terminated = self._compute_reward()

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()

        print(f"Terminated: {terminated} | Block steady: {self.has_block_steady} | Valid Flip: {self.valid_flip}")

        return observation, reward, terminated, False, info


    def _wait_for_block_to_land(self):
        while not self.has_block_steady:
            mujoco.mj_step(self.model, self.data)
            if has_block_released(self.data):
                self.has_block_released = True
                if has_block_hit_robot(self.data):
                    self.valid_flip = False
                    break
            if has_block_landed(self.data):
                if self.trigger_iteration == 0:
                    self.has_block_landed = True
                    self.block_landing_quat = get_block_pose(self.model, self.data, quat=True)[1]
                    self.trigger_iteration += 1
            if np.linalg.norm(self.data.sensor('block_linvel').data.copy()) < 0.001 and np.linalg.norm(self.data.sensor('block_angvel').data.copy()) < 0.001:
                self.block_steady_quat = get_block_pose(self.model, self.data, quat=True)[1]
                self.has_block_steady = True


    def _compute_reward(self):
        block_height = self.block_position[2]
        desired_block_landing_quat = [0, 0, -1, 0]

        if not self.has_gripper_opened: 
            if has_block_hit_floor(self.data):
                print("Block hit floor before flipping")
                return -20, True
            
            elif self._is_close_to_desired_state():
                return 20, False

            else:
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
                return reward, False            

        else:
            # Blocks hit robot
            if not self.valid_flip:
                print("Block hit robot")
                return -10, True
            
            else:
                if has_block_flipped(self.initial_block_quat, self.block_steady_quat):
                    print("Block flipped")
                    return 50, True
                else:
                    orientation_error = quat_distance_new(desired_block_landing_quat, self.block_landing_quat)
                    print("Block orientation error")
                    return -orientation_error, True


    def _get_obs(self): 
        joint_pos = self.data.qpos[:6].copy() 
        joint_vel = self.data.qvel[:6].copy()  
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

        joint_pos = obs[:6]
        joint_vel = obs[6:12]
        translational_velocity_residual, angular_velocity_residual, height_residual, orientation_residual = self._calculate_residuals()

        info = {
            "mpc_timestep": self.mpc_timestep,
            "joint_positions": joint_pos.tolist(),
            "joint_velocities": joint_vel.tolist(),
            "ee_height_residual": height_residual,
            "translational_velocity_residual": translational_velocity_residual,
            "angular_velocity_residual": angular_velocity_residual,
            "orientation_residual": orientation_residual,
            "block_landed": self.has_block_landed,
            "gripper_opened": self.has_gripper_opened,
            "valid_flip": self.valid_flip,
        }
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
#     num_episodes = 5  # Number of episodes to test
#     print(f"Starting manual test with {num_episodes} episodes...")

#     for episode in range(num_episodes):
#         observation, info = env.reset()
#         print(f"--- Episode {episode + 1}/{num_episodes} ---")
#         print("Initial observation:", observation)
#         print("Action space (low):", env.action_space.low)
#         print("Action space (high):", env.action_space.high)

#         total_reward = 0
#         terminated = False
#         action_count = 0

#         while not terminated:
#             action = env.action_space.sample()  # Generate a random action
#             observation, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
#             action_count += 1

#             # Print step information
#             print(f"Step {action_count}:")
#             print(f"Action: {action}")
#             print(f"Observation: {observation}")
#             print(f"Reward: {reward}")
#             print(f"Terminated: {terminated}")

#         print(f"--- Episode {episode + 1} Summary ---")
#         print(f"Total reward: {total_reward}")
#         print(f"Total steps: {action_count}")
#         print(f"Block pose: {get_block_pose(env.model, env.data, 'block_0')}")
#         print(f"Gripper opened: {env.has_gripper_opened}")

#     env.close()
#     print("Manual test completed.")


# if __name__ == "__main__":
#     manual_test()

#     env = URFlipBlockEnv(render_mode='human')
#     check_env(env, skip_render_check=False)

