import research_main

import argparse
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
from mujoco_env import make_mujoco_env

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="research_main/FlipBlock-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-path", type=str, default="policy results/best_policy.pth")
    return parser.parse_args()

def evaluate_td3(args=get_args()):

    env, _, _ = make_mujoco_env(
        args.task,
        args.seed,
        num_train_envs=1,
        num_test_envs=args.test_num,
        obs_norm=False,
    )


    env = gym.make(args.task, render_mode="human")  

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

if __name__ == "__main__":
    evaluate_td3()
