import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.spaces.utils import flatten_space, flatten, unflatten

from research_main.envs.utils import *
from research_main.envs.scene_builder import *
from research_main.envs.finite_state_machine import *

#from utils import *
#from scene_builder import *
#from finite_state_machine import *


class URFlipBlockEnv(gym.Env):
    metadata = {"render_modes": ["livecam", "video"], "contact_vis": [True, False]}

    def __init__(self, render_mode=None, contact_vis=None):
        block_positions_orientations = [([0.5, 0.5, 0.1], [np.pi/2, 0, 0])]
        world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations)
        
        self.model = mujoco.MjModel.from_xml_string(world_xml_model)
        self.data = mujoco.MjData(self.model)
        self.contact = mujoco.MjContact()

        mujoco.mj_kinematics(self.model, self.data)

        self.render_mode = render_mode
        self.contact_vis = contact_vis
        self.fsm = FiniteStateMachine(self.model)
        
        if self.render_mode is not None:
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self.camera)
            
            if self.render_mode == 'livecam':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            elif self.render_mode == 'video':
                self.renderer = mujoco.Renderer(self.model, height=1024, width=1440)
                self.framerate = 60
                self.frames = []

        if self.contact_vis is not None:
            self.options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.options)
            self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

            # tweak scales of contact visualization elements
            self.model.vis.scale.contactwidth = 0.1
            self.model.vis.scale.contactheight = 0.03
            self.model.vis.scale.forcewidth = 0.05
            self.model.vis.map.force = 0.3

        # Observation space: joint positions (2), joint velocities (2), block orientation (3)
        #Action space: elbow velocity, wrist 1 velocity, release
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
        self.action_space = spaces.MultiDiscrete([21, 21, 2])
        #self.action_space = spaces.Dict({
        #    "elbow_velocity": spaces.Box(low=-np.pi, high=0, shape=(1,), dtype=np.float64),
        #    "wrist_1_velocity": spaces.Box(low=-np.pi, high=0, shape=(1,), dtype=np.float64),
        #    "release": spaces.MultiBinary(1)
        #})
        #self.action_space = flatten_space(original_action_space)
        #print("Action space:", self.action_space)
        
        #self.original_action_space = original_action_space
        self.has_block_landed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.has_block_landed = False
        self.fsm.has_gripper_opened = False
        reset_block_position(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)        

        if self.render_mode == 'video':
            self.frames = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(f'gym_test_{time.time()}.mp4', fourcc, self.framerate, (1440, 1024))

        while self.fsm.state not in ['flip_block', 'move_back']:
            current_position, _ = get_ee_pose(self.model, self.data)
            self.fsm.state = self.fsm.reset_pose(self.model, self.data, current_position)
            mujoco.mj_step(self.model, self.data)
            
            if self.render_mode is not None:
                self.render()
        
        self.force_reset = False
        self.block_orientation_hist = []
        self.block_contact_hist = np.empty((0, 2))
        self.block_height_hist = []
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _compute_reward(self):
        reward = 0  

        #if self.fsm.has_gripper_opened == False:
        #    print("Gripper not opened.")
        #    return -300

        if has_hit_robot(self.data, self.block_contact_hist):
            #print("Robot hit by block.")
            reward += -150

        block_position, _ = get_block_pose(self.model, self.data, 'block_0')
        if block_position[0] < 0:
            #print("Block went out of bounds.")
            reward += -200

        if reward == 0:
            if has_rotated(self.block_orientation_hist) and has_flipped(self.data):
                #print("Block successfully flipped.")
                reward += 100
            elif has_flipped(self.data):
                #print("Block flipped but not rotated.")
                reward += -50
            elif has_rotated(self.block_orientation_hist):
                #print("Block rotated but not flipped.")
                reward += 30
            else:
                #print("Block not flipped or rotated.")
                reward += -100

        return reward

    def step(self, action):
        #print("Action: ", action)
        #original_action = unflatten(self.original_action_space, action)
        terminated = True if(action[2] == 1 or self.data.time > 8) else False

        self.fsm.do_flip(self.model, self.data, action)
        mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        if terminated:
            if self.fsm.has_gripper_opened == True: 
                self._wait_for_block_to_land()
                reward = self._compute_reward()
            else:
                # The robot does not release the block until the episode ends 
                self.fsm.state = 'post_flip_block'
                reward = -300

        else:
            reward = 0
        
        if self.render_mode is not None:
            self.render()
        
        if terminated and self.render_mode == 'video':
            self.out.release()

        return observation, reward, terminated, False, info

    def _wait_for_block_to_land(self):
        while self.has_block_landed == False:
            self.fsm.do_move_back(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if self.render_mode is not None:
                self.render()
            
            # Conditional to check  the block state after being released
            if self.fsm.has_block_released:
                block_position, block_orientation = get_block_pose(self.model, self.data, 'block_0')
                self.block_orientation_hist.append(block_orientation)
            
                # To check whether the block hit the robot or not
                if len(self.block_height_hist) < 4:
                    self.block_height_hist.append(block_position[2])
                if len(self.block_height_hist) == 3:
                    if is_block_declining(self.block_height_hist):
                        new_contacts = detect_block_contact(self.data)
                        if new_contacts:
                            self.block_contact_hist = np.vstack([self.block_contact_hist, new_contacts])

                # To check whether the block has landed or not                                        
                block_trans_velocity, block_ang_velocity = get_block_velocity(self.data)
                if block_trans_velocity < 0.01 and block_ang_velocity < 0.001:
                    self.has_block_landed = True

    def render(self):
        if self.render_mode == 'livecam':
            self.viewer.sync()
        elif self.render_mode == 'video':
            if len(self.frames) < self.data.time * self.framerate:
                self.camera.distance = 4
                self.camera.azimuth = 150
                self.camera.elevation = -30
                if self.contact_vis is not None:
                    self.renderer.update_scene(self.data, self.camera, self.options)
                else:
                    self.renderer.update_scene(self.data, self.camera)
                pixels = self.renderer.render()
                self.frames.append(pixels)
                self.out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.render_mode == 'video' and hasattr(self, 'out'):
            self.out.release()

    def _get_obs(self): 
        # Observation space: joint positions (2), joint velocities (2), block orientation (3)
        self.elbow_id = self.model.joint('elbow_joint').id
        self.wrist_1_id = self.model.joint('wrist_1_joint').id 
        
        self.active_joint_ids = [self.elbow_id, self.wrist_1_id]
        self.active_joint_angles = get_specific_joint_angles(self.data, self.active_joint_ids)
        self.active_joint_velocities = get_specific_joint_velocities(self.data, self.active_joint_ids)

        _, self.block_orientation = get_block_pose(self.model, self.data, 'block_0')
       
        obs = np.concatenate([self.active_joint_angles, self.active_joint_velocities, self.block_orientation])
        
        return obs

    def _get_info(self):
        return {
            "state": self.fsm.state,
            "joint_angles": self.active_joint_angles,
            "joint_velocities": self.active_joint_velocities,
            "block_orientation": self.block_orientation,
            "block_landed": self.has_block_landed,
        }

    def seed(self, seed):
        np.random.seed(seed)

def manual_test():
    def get_action(env, observation):
        release_elbow_angle = 0.9
        elbow_angle = observation[0] 

        if env.fsm.has_block_released == False:
            if elbow_angle < release_elbow_angle:
                if episode % 2 != 0:
                    action = [0, 0, 0] # Tes kalo baloknya ga dilepas gimana (khusus episode genap)
    
                else:
                    action = [20, 20, 1]
            else:
                action = [20, 20, 0] # Mengayunkan joints
        else:
            action = [0, 0, 1]
        return action

    num_episodes = 1
    episode_rewards = []

    env = URFlipBlockEnv(render_mode='livecam') 

    for episode in range(num_episodes):
        observation, info = env.reset()
        print(f"Episode {episode + 1}/{num_episodes}")
        #print("Initial info:", info)
        print("Id of subblock blue: ", env.data.geom('blue_subbox').id)
        print("Id of subblock orange: ", env.data.geom('orange_subbox').id)
        print("Id of floor: ", env.data.geom('floor').id)

        total_reward = 0
        print_iteration = 0
        terminated = False

        while not terminated:
            action = get_action(env, observation)

            observation, reward, terminated, truncated, info = env.step(action)

            print("Is terminated: ", terminated, "Observation: ", observation, "Reward: ", reward, "Action: ", action)

            if env.fsm.has_block_released == True and print_iteration < 5:
                print("Is terminated: ", terminated, "Observation: ", observation, "Reward: ", reward, "Action: ", action)
                print_iteration += 1

            total_reward += reward

        
        #for contact in env.block_contact_hist:
        #    print("Contact between:", contact[0], contact[1])
        print("Block pose: ", get_block_pose(env.model, env.data, 'block_0'))
        print("Has gripper opened: ", env.fsm.has_gripper_opened)
        print("Resetting environment for next episode.")
        
    env.close()

    print("Total rewards:", total_reward)   
    # Print the rewards for all episodes
    #print("Action calls:", action_call)
    #print("Episode rewards:", episode_rewards)
    #print("Average reward over episodes:", np.mean(episode_rewards))


if __name__ == "__main__":
    #env = URFlipBlockEnv()
    #check_env(env, skip_render_check=False)
    #print(env.action_space.sample())
    manual_test()