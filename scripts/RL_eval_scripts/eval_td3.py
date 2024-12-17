import research_main

import os
import argparse
import json
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector
from tianshou.exploration import GaussianNoise
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from ..RL_train_scripts.mujoco_env import make_mujoco_env


from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def evaluate_td3(policy_version, policy_type):
    class Args:
        def __init__(self, policy_version, policy_type):
            self.task = "research_main/FlipBlock-v0"
            self.seed = 0
            self.hidden_sizes = [256, 256]
            self.actor_lr = 3e-4
            self.critic_lr = 3e-4
            self.gamma = 0.99
            self.tau = 0.005
            self.exploration_noise = 0.1
            self.policy_noise = 0.2
            self.noise_clip = 0.5
            self.update_actor_freq = 2
            self.n_step = 1
            self.test_num = 1
            self.device = "cpu"
            self.render_modes = "cpu"
            self.policy_version = policy_version
            self.policy_type = policy_type

    args = Args(policy_version, policy_type)


    # env, _, _ = make_mujoco_env(
    #     args.task,
    #     args.seed,
    #     num_train_envs=1,
    #     num_test_envs=args.test_num,
    #     obs_norm=False,
    # )
    if policy_version:
        args.policy_version = policy_version
    if policy_type:
        args.policy_type = policy_type

    base_dir = os.path.dirname(os.path.abspath(__file__))

    policy_base_dir = os.path.join(base_dir, "..", "..", "policy results")
    args.load_path = os.path.join(policy_base_dir, args.policy_version, f"{args.policy_type}.pth")

    env = gym.make(
        args.task,
        use_mode="RL_eval",
        policy_version=args.policy_version,
        policy_type=args.policy_type
    )

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.policy_noise = args.policy_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define the policy network structures
    net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net_a, args.action_shape, max_action=args.max_action, device=args.device).to(
        args.device,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy: TD3Policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )


    policy.load_state_dict(torch.load(args.load_path, map_location=torch.device('cpu')))


    #test_envs.seed(args.seed)
    # rec_env.seed(args.seed)

    #test_collector = Collector(policy, test_envs)
    test_collector = Collector(policy, env)
    
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num)
    print(collector_stats)


# if __name__ == "__main__":
#     evaluate_td3(policy_version, policy_type)
