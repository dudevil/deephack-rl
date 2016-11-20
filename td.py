import gym
import numpy as np
import argparse
from collections import OrderedDict
np.set_printoptions(precision=2, suppress=True)


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


def temporal_difference(env, value, policy, num_episodes=5000, gamma=0.99, alpha=.05):
    for ep in range(num_episodes):
        done = False
        prev_s = s = env.reset()
        while not done:
            action = policy(s)
            s, reward, done, info = env.step(action)
            prev_v = value[prev_s]
            value[prev_s] = prev_v + alpha * (reward + gamma * value[s] - prev_v)
            prev_s = s

        if ep % 1000 == 0:
            print("episode {0}...".format(ep))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=5000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
                        help='Gamma discount factor')
    parser.add_argument('--alpha', '-a', metavar='a', type=float, default=0.05, nargs='?',
                        help='Alpha parameter')
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.env == "FrozenLake8x8-v0":
        policy = FrozenLakeOptimalPolicy(env)

    value = ValueFuntion(env)
    value = temporal_difference(env, value, policy, num_episodes=args.num_episodes, gamma=args.gamma)
    print(value)
