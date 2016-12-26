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

    def update_e(self):
        self.counter += 1
        self.e = 0.999985 ** self.counter

    def __call__(self, s):
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

    def update_e(self):
        pass


def sarsa_lambda(env, q, policy, num_episodes=5000, gamma=0.99, alpha=0.005, lmbda=0.9):
    rewards = []

    for ep in range(num_episodes):
        done = False
        R = 0
        prev_s = s = env.reset()
        z = defaultdict(float)
        while not done:
            action = policy(s)
            s, reward, done, info = env.step(action)
            z[(prev_s, action)] += 1
            delta = reward + gamma * q[s][policy(s)] - q[prev_s][action]
            for s_, a_ in z.keys():
                q[s_][a_] += alpha * delta * z[s_, a_]
                z[s_, a_] *= gamma * lmbda
            prev_s = s
            R += reward

        policy.update_e() # decay epsilon once per episode
        
        rewards.append(R)
        if ep % 1000 == 0:
            print("[episode {0}] Avg reward: {1:.3f} Epsilon: {2:.3f}...".format(ep, np.mean(rewards[-1000:]), policy.e))
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sarsa-lambda algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=500000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
                        help='Gamma discount factor')
    parser.add_argument('--alpha', '-a', metavar='a', type=float, default=0.05, nargs='?',
                        help='Alpha parameter')
    parser.add_argument('--lmbda', '-l', metavar='l', type=float, default=0.5, nargs='?',
                        help='Lambda parameter')
    args = parser.parse_args()

    env = gym.make(args.env)
    q = {}
    for s in range(env.nS):
        q[s] = np.zeros(env.action_space.n)

    policy = EgreedyPolicy(env, q, e=1.)
    policy = sarsa_lambda(env, q, policy, num_episodes=args.num_episodes,
                          gamma=args.gamma, alpha=args.alpha, lmbda=args.lmbda)

    policy = GreedyPolicy(env, q)

    env.monitor.start('%s-sarsa-lambda-1' % args.env, force=True)

    for i_episode in range(400000):
        s = env.reset()
        done = False
        while not done:
            #env.render()
            action = policy(s)
            s, reward, done, info = env.step(action)
    env.monitor.close()
