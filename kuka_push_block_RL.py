import research_main

import os
import gymnasium as gym
import tianshou as ts
import torch, numpy as np
from torch import nn

from tianshou.data import Batch
from tianshou.policy.base import BasePolicy
from tianshou.exploration import GaussianNoise

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import TensorboardLogger

device = "cpu"

# Make an envurioment
env = gym.make("research_main/FlipBlock-v0")

# Setup vectorized environments
train_envs = ts.env.DummyVectorEnv([lambda: gym.make('research_main/FlipBlock-v0') for _ in range(1)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('research_main/FlipBlock-v0') for _ in range(1)])

# Set up the network
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
hidden_sizes = [128, 128]
actor_lr = 1e-3
critic_lr = 1e-3
max_action = env.action_space.high[0]

print("Max action:", max_action)
print("Observations shape:", state_shape)
print("Actions shape:", action_shape)
print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

np.random.seed(0)
torch.manual_seed(0)

net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes, device=device)
actor = Actor(net_a, action_shape, hidden_sizes, max_action=max_action, device=device).to(device,)
actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
net_c = Net(
    state_shape=state_shape,
    action_shape=action_shape,
    hidden_sizes=hidden_sizes,
    concat=True,
    device=device,
)
critic = Critic(net_c, device=device).to(device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

# Set up policy
policy = ts.policy.DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        exploration_noise=GaussianNoise(sigma=0.1),
        estimation_step=5,
        action_space=env.action_space,
    )

# Set up the collector
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 1), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

#writer = SummaryWriter('log/dqn')
#logger = TensorboardLogger(writer)    

writer = SummaryWriter('log/ddpg5')
logger = TensorboardLogger(writer)

print("Start training")

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=5, step_per_epoch=2000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=128,
    logger=logger,
    test_in_train=False,
    verbose=True,
    ).run()

print('Finished training!')

# Save policy
torch.save(policy.state_dict(), 'ddpg5.pth')
policy.load_state_dict(torch.load('ddpg5.pth'))


policy.eval()
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=0)