import research_main
import tianshou as ts
import gymnasium as gym
import torch, numpy as np
from torch import nn

from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

device="cpu"

# Make an envurioment
env = gym.make("research_main/PushBlock-v0", render_mode="GUI")

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
hidden_sizes = [128, 128]
actor_lr = 1e-3
critic_lr = 1e-3
max_action = env.action_space.high[0]

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


policy = ts.policy.DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        exploration_noise=GaussianNoise(sigma=0.1),
        estimation_step=5,
        action_space=env.action_space,
    )

policy.load_state_dict(torch.load('ddpg.pth'))

policy.eval()
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=3, render=0)