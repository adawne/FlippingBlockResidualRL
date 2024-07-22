import research_main

import argparse
import pprint
import gymnasium as gym
import tianshou as ts
import torch, numpy as np

from os.path import join
from torch import nn
from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.gym_wrappers import MultiDiscreteToDiscrete
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='FlippingEnv-v0')
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--train-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--hidden-layers', type=int, default=2, help='Number of hidden layers in the network')
    parser.add_argument('--units-per-layer', type=int, default=128, help='Number of units per hidden layer')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--eps-test', type=float, default=0)
    parser.add_argument('--eps-decay-steps', type=int, default=40000, help='Number of steps over which epsilon decays')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--prioritized-replay', default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.0)
    parser.add_argument(
        '--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    args = parser.parse_known_args()[0]

    print("Arguments:", vars(args))
    return args



def train_dqn(args=get_args()):
    # Create the wrapped environment
    env = gym.make("research_main/FlipBlock-v0")

    # Make the training and testing environments
    train_envs = SubprocVectorEnv([lambda: gym.make("research_main/FlipBlock-v0") for _ in range(args.train_num)])
    test_envs = SubprocVectorEnv([lambda: gym.make("research_main/FlipBlock-v0") for _ in range(args.test_num)])

    # Build the network
    class Net(nn.Module):
        def __init__(self, state_shape, action_shape, hidden_layers, units_per_layer):
            super().__init__()
            layers = []
            input_size = np.prod(state_shape)
            
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_size, units_per_layer))
                layers.append(nn.ReLU(inplace=True))
                input_size = units_per_layer
            
            layers.append(nn.Linear(input_size, np.prod(action_shape)))
            self.model = nn.Sequential(*layers)

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, args.hidden_layers, args.units_per_layer)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space= env.action_space,
        discount_factor=0.9,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs), alpha=args.alpha, beta=args.beta, ignore_obs_next=True,)
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True)


    train_collector = ts.data.Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_step=args.batch_size * args.train_num)

    logger = WandbLogger(
        train_interval=1,
        update_interval=1,
        info_interval=1,
        test_interval=1,
        project=args.task,
        config=args,
    )

    log_path = join(args.logdir, args.task)
    writer = SummaryWriter(log_path)
    logger.load(writer)

    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 10000 + args.eps_decay_steps:
            eps = args.eps_train - (env_step - 10000) / args.eps_decay_steps * (0.9 * args.eps_train)
            policy.set_eps(max(eps, 0.1 * args.eps_train))  # Ensure epsilon doesn't go below 10% of initial epsilon
        else:
            policy.set_eps(0.1 * args.eps_train)


        if args.prioritized_replay:
            if env_step <= 10000:
                beta = args.beta
            elif env_step <= 50000:
                beta = args.beta - (env_step - 10000) / 40000 * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buf.set_beta(beta)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_best_fn(policy):
        save_path = join(args.logdir, args.task, 'best_policy.pth')
        torch.save(policy.state_dict(), save_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        save_path = join(args.logdir, args.task, f'checkpoint_epoch{epoch}_policy.pth')
        torch.save(policy.state_dict(), save_path)

        return save_path


    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch, step_per_epoch=args.step_per_epoch, step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step, episode_per_test=100, batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        save_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        verbose=True,
        logger=logger
    ).run()

    torch.save(policy.state_dict(), 'Flipping_DQN.pth')
    
    
    if __name__ == "__main__":
        pprint.pprint(result)
        env = DummyVectorEnv([lambda: gym.make("research_main/FlipBlock-v0")])
        policy.load_state_dict(torch.load('Flipping_DQN.pth'))
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = ts.data.Collector(policy, env)
        result = collector.collect(n_episode=args.test_num)
        rews, lens = result.returns, result.lens
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

if __name__ == "__main__":
    train_dqn() 
