#!/usr/bin/env python3
import gym
import argparse
import numpy as np
from collections import OrderedDict


class ValueFuntion(OrderedDict):

    def __init__(self, env, *args, **kwargs):
        self.env = env
        super(ValueFuntion, self).__init__(*args, **kwargs)
        for s in range(self.env.nS):
            self[s] = 0

    def __str__(self, *args, **kwargs):
        if hasattr(env, 'ncol') and hasattr(env, 'nrow'):
            return np.array_str(np.array(list(self.values())).reshape(env.ncol, env.nrow),
                                precision=2, suppress_small=True)
        return super(ValueFuntion, self).__str__(*args, **kwargs)


class TablePolicy(OrderedDict):

    def __init__(self, env):
        self.env = env
        for s in range(self.env.nS):
            self[s] = self.env.action_space.sample()

    def __str__(self, *args, **kwargs):
        if isinstance(self.env, gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
            mapping = {
                0: '<',
                1: 'v',
                2: '>',
                3: '^'
            }
            result = np.array_str(np.array([mapping[policy(s)] if env.desc.flatten()[s] != b'H' else 'H'
                                            for s in range(env.nS)]).reshape(env.ncol, env.nrow),
                                  precision=2, suppress_small=True)
            return result
        return super(TablePolicy, self).__str__(*args, **kwargs)

    def __call__(self, s):
        return self[s]


class FrozenLakeOptimalPolicy(OrderedDict):

    def __init__(self, env):
        self.env = env
        optimal_policy = [3, 2, 2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 2, 2, 1,
                         3, 3, 0, 3, 2, 3, 2, 1,
                         3, 3, 3, 1, 0, 2, 2, 2,
                         0, 3, 0, 3, 2, 1, 3, 2,
                         0, 0, 2, 1, 3, 0, 3, 2,
                         0, 1, 1, 0, 2, 0, 1, 2,
                         0, 1, 0, 3, 1, 2, 1, 1]

        for s in range(self.env.nS):
            self[s] = optimal_policy[s]

    def __str__(self, *args, **kwargs):
        if isinstance(self.env, gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
            mapping = {
                0: '<',
                1: 'v',
                2: '>',
                3: '^'
            }
            result = np.array_str(np.array([mapping[policy(s)] if env.desc.flatten()[s] != b'H' else 'H'
                                            for s in range(env.nS)]).reshape(env.ncol, env.nrow),
                                  precision=2, suppress_small=True)
            return result
        return super(FrozenLakeOptimalPolicy, self).__str__(*args, **kwargs)

    def __call__(self, s):
        return self[s]


def value_iteration(env, value, policy, gamma, theta=1e-3):
    delta = np.inf
    while delta > theta:
        delta = 0
        for s in range(env.nS):
            a = policy(s)
            nv = sum([p * (r + gamma * value[next_s]) * (not done)
                      for p, next_s, r, done
                      in env.P[s][a]])
            delta = max(delta, abs(value[s] - nv))
            value[s] = nv


       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Value Iteration algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--policy', '-p', type=str, default='', nargs='?',
                        help='Policy file to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=1000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
                        help='Gamma discount factor')
    args = parser.parse_args()

    env = gym.make(args.env)

    value = ValueFuntion(env)
    
    if args.policy:
        policy = TablePolicy(env)
        with open(args.policy, 'r') as f:
            for line in f.readlines():
                s, a = line.split()
                policy[int(s)] = int(a)

        print("Read policy from file {}".format(args.policy))
        print(policy)
    else:
        assert args.env == "FrozenLake8x8-v0"
        policy = FrozenLakeOptimalPolicy(env)

    value_iteration(env, value, policy, gamma=args.gamma)
    print(policy)
    print(value)

    env.monitor.start('%s-value-iteration-1' % args.env, force=True)

    ep_rewards = []
    for ep in range(args.num_episodes):
        done = False
        R = 0
        s = env.reset()
        while not done:
            #env.render()
            action = policy[s]
            s, reward, done, info = env.step(action)
            R += reward
        ep_rewards.append(R)

    env.monitor.close()
    print("Avg rewards over {0} episodes: {1:.3f} +/-{2:.3f}".format(args.num_episodes, np.mean(ep_rewards), np.std(ep_rewards)))
