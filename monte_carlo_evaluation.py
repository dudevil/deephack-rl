import gym
import argparse
import numpy as np
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


def monte_carlo(env, value, policy, num_episodes=10000, gamma=0.99):
    visit_counter = {}

    for ep in range(num_episodes):
        done = False
        s = env.reset()
        visited = [(s, 0)]
        visit_counter[s] = visit_counter.get(s, 0) + 1

        # play an episode
        while not done:
            action = policy(s)
            s, reward, done, info = env.step(action)
            visited.append((s, reward))
            visit_counter[s] = visit_counter.get(s, 0) + 1
        # calculate the reward
        G = 0.
        for s, r in reversed(visited):
            G = gamma * G + r
            count = float(visit_counter[s])
            # based on the propery of the mean
            # mu_k = 1/k * [(k - 1) * mu_k-1 + x_k]
            value[s] = ((count - 1) * value[s] + G) / count

        if ep % 1000 == 0:
            print("[episode {0}] Start return: {1:.2f}".format(ep, G))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=25000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
                        help='Gamma discount factor')
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.env == "FrozenLake8x8-v0":
        policy = FrozenLakeOptimalPolicy(env)

    value = ValueFuntion(env)
    value = monte_carlo(env, value, policy, num_episodes=args.num_episodes, gamma=args.gamma)
    print(value)
