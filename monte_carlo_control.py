import gym
import argparse
import numpy as np
from collections import defaultdict

np.set_printoptions(precision=2, suppress=True)


class EgreedyPolicy:

    def __init__(self, env, q, e=0.9):
        self.env = env
        self.q = q
        self.e = e
        self.counter = 0

    def _update_e(self):
        self.e = 0.9999999 ** self.counter

    def __call__(self, s):
        self._update_e()
        self.counter += 1
        if np.random.sample() < self.e:
            return self.env.action_space.sample()
        else:
            q = self.q[s]
            return np.random.choice(np.where(q == q.max())[0])  # ties are broken at random

    def __str__(self, *args, **kwargs):
        if isinstance(self.env, gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
            mapping = {
                0: '<',
                1: 'v',
                2: '>',
                3: '^'
            }
            e_backup = self.e
            result = np.array_str(np.array([mapping[policy(s)] if env.desc.flatten()[s] != b'H' else 'H'
                                            for s in range(env.nS)]).reshape(env.ncol, env.nrow),
                                  precision=2, suppress_small=True)
            self.e = e_backup
            return result
        return super(EgreedyPolicy, self).__str__(*args, **kwargs)


def mc_contol(env, q, policy, num_episodes=100, gamma=0.99):
    returns = defaultdict(float)
    N = defaultdict(float)
    num_wins = 0
    for ep in range(num_episodes):
        done = False
        visited = []
        s = env.reset()
        # play an episode
        while not done:
            action = policy(s)
            next_s, reward, done, info = env.step(action)
            visited.append((s, reward, action))
            s = next_s
        # get reward and update state-actions
        G = 0
        for s, r, a in reversed(visited):
            G = gamma * G + r
            returns[(s, a)] += G
            N[(s, a)] += 1.
            q[s][a] = returns[(s, a)] / N[(s, a)]

        if G > 0:
            num_wins += 1
        if ep % 1000 == 0:
            print("[episode {0}] # of wins: {1} Epsilon: {2:.3f}...".format(ep, num_wins, policy.e))
            num_wins = 0
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sarsa-lambda algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=100000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
                        help='Gamma discount factor')
    args = parser.parse_args()

    env = gym.make(args.env)
    q = {}
    for s in range(env.nS):
        q[s] = np.zeros(env.action_space.n)

    policy = EgreedyPolicy(env, q, e=1.)
    policy = mc_contol(env, q, policy, num_episodes=args.num_episodes, gamma=args.gamma)
    policy.e = 0.
    print(policy)

