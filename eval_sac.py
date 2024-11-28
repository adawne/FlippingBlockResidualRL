import argparse
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from mujoco_env import make_mujoco_env

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-path", type=str, default="log/Ant-v4/policy.pth")
    return parser.parse_args()

def evaluate_sac(args=get_args()):

    env, _, test_envs = make_mujoco_env(
        args.task,
        args.seed,
        num_train_envs=1,
        num_test_envs=args.test_num,
        obs_norm=False,
    )


    env = gym.make("Ant-v4", render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="agent_eval", name_prefix="eval",
                    episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    rec_env = SubprocVectorEnv(
        [
            lambda: env
        ]
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define the policy network structures
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        conditioned_sigma=True,
        device=args.device
    ).to(args.device)

    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)

    # Optimizers are not needed for evaluation but required by the policy
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    policy.load_state_dict(torch.load(args.load_path))

    #test_envs.seed(args.seed)
    rec_env.seed(args.seed)

    #test_collector = Collector(policy, test_envs)
    test_collector = Collector(policy, rec_env)
    
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num)
    print(collector_stats)

if __name__ == "__main__":
    evaluate_sac()
