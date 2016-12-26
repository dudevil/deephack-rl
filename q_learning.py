#!/usr/bin/env python3
import gym
import numpy as np
import argparse
np.set_printoptions(precision=2, suppress=True)


class EgreedyPolicy:

    def __init__(self, env, q, e=0.9):
        self.env = env
        self.q = q
        self.e = e
        self.counter = 0

    def _update_e(self):
        self.e = 0.99999 ** self.counter

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

class GreedyPolicy(EgreedyPolicy):

    def __init__(self, env, q):
        super(GreedyPolicy, self).__init__(env, q, e=0.)

    def _update_e(self):
        pass

    
def q_learning(env, q, policy, num_episodes=100, gamma=0.99, alpha=0.05):
    rewards = []
    for ep in range(num_episodes):
        done = False
        prev_s = s = env.reset()
        R = 0
        while not done:
            action = policy(s)
            s, reward, done, info = env.step(action)
            q[prev_s][action] += alpha * (reward + gamma * q[s].max() - q[prev_s][action])
            prev_s = s
            R += reward  # for bookkeeping
        rewards.append(R)

        if ep % 1000 == 0 and ep > 0:
            print("[{0}] Avg reward: {1:.3f} epsilon: {2:.4f}".format(ep, np.mean(rewards[-1000:]), policy.e))
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?', # FrozenLake8x8-v0
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=50000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.995, nargs='?',
                        help='Gamma discount factor')
    parser.add_argument('--alpha', '-a', metavar='a', type=float, default=0.05, nargs='?',
                        help='Alpha parameter')
    args = parser.parse_args()

    env = gym.make(args.env)
    q = {}
    for s in range(env.nS):
        q[s] = np.zeros(env.action_space.n)

    policy = EgreedyPolicy(env, q, e=1.)
    _ = q_learning(env, q, policy, num_episodes=args.num_episodes, gamma=args.gamma, alpha=args.alpha)

    env = gym.make(args.env)
    env.monitor.start('%s-qlearning-1' % args.env, force=True)
    
    policy = GreedyPolicy(env, q)
    rewards  = []
    for i_episode in range(1000):
        s = env.reset()
        done = False
        R = 0.
        while not done:
            #env.render()
            action = policy(s)
            s, reward, done, info = env.step(action)
    env.monitor.close()
