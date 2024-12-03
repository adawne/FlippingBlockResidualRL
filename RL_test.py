import gymnasium as gym
import numpy as np
import research_main
from gymnasium.wrappers import RescaleAction, NormalizeReward, NormalizeObservation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.utils.env_checker import check_env


def manual_test():
    env = gym.make("research_main/FlipBlock-v0")
    # env = RecordVideo(env, video_folder="agent_eval", name_prefix="eval",
    #                 episode_trigger=lambda x: True)
    # env = RecordEpisodeStatistics(env)

    # env = gym.make("Pusher-v4")


    num_episodes = 1
    max_steps_per_episode = 1000

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        observation, info = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Sample a random action
            action = env.action_space.sample()
            #action = np.array([0, -0.2, -0.2, -0.5, 0, 0])
            
            print(f"Action outside: {action}")
            # Take a step in the environment with the random action
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Qpos: {observation[:6]} | Qvel: {observation[6:12]} | EE height: {observation[12]} | EE quat: {observation[13:17]} | EE transvel: {observation[17:20]} | EE angvel: {observation[20:23]}") 
            print(f"Reward outside: {reward}")
            total_reward += reward

            # Print the action, observation, and reward
            print(f"Step {step + 1}")
            
    
            # # Check if the episode is terminated or truncated
            if terminated or truncated:
                print(f"Episode finished after {step + 1} steps with total reward: {total_reward}")
                break



if __name__ == "__main__":
    # manual_test()

    env = gym.make("research_main/FlipBlock-v0")
    # env = RescaleAction(env, min_action=-1, max_action=1)
    # env = NormalizeObservation(env)
    # env = NormalizeReward(env, gamma=0.99)

    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high
    min_action = env.action_space.low

    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)

    # print(action_shape, max_action, min_action)