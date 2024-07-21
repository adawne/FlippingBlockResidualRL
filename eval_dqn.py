import research_main
import argparse
import numpy as np
import torch
import gymnasium as gym
import tianshou

from torch import nn
from tianshou.data import Batch, to_numpy
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--eps-test', type=float, default=0)
    args = parser.parse_known_args()[0]
    return args


def load_policy(args, render_mode=None):

    env = gym.make("research_main/FlipBlock-v0", render_mode=render_mode)

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # seed
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    # env.seed(args.seed)

    # Build the network
    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
            )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    print("Action shape:", action_shape)
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)


    policy = tianshou.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space= env.action_space,
        discount_factor=0.9,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    policy.load_state_dict(torch.load('checkpoint_epoch199_policy.pth'))
    policy.eval()
    policy.set_eps(args.eps_test)

    return env, policy

def run_base_eval(env, policy, verbose=False):
    # use the eval policy to interact with the env
    # https://tianshou.org/en/stable/01_tutorials/00_dqn.html#setup-collector
    length, reward = 0, 0
    batch = Batch(
        obs={},
        act={},
        rew={},
        terminated={},
        truncated={},
        done={},
        info={},
    )
    obs, info = env.reset()
    batch.update(obs=obs, info=info)

    while True:
        result = policy(batch)
        act = to_numpy(result.act)
        #print("Batch: ", batch, "Result: ", result, "Act: ", act)
        obs_next, rew, terminated, truncated, info = env.step(act)
        length += 1
        reward += rew
        done = np.logical_or(terminated, truncated)
        env.render()

        # assembly continuing
        if not done:
            batch.update(
                act=act,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            batch.obs = obs_next
        else:
            break   
            

    return length, reward

def simulate_flipping(args=get_args()):
    env, policy = load_policy(args, render_mode='livecam')
    length, reward = run_base_eval(env, policy)
    print(f"Length: {length}, Reward: {reward}")

if __name__ == "__main__":
    simulate_flipping()