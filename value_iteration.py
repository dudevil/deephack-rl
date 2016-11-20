#!/usr/bin/env python
import gym
import numpy as np
import argparse
from collections import OrderedDict
env = gym.make('FrozenLake8x8-v0')

np.set_printoptions(precision=2, suppress=True)

value = OrderedDict()


def v_pp():
    print(np.array(list(value.values())).reshape(env.ncol, env.nrow))


def p_pp(policy):
    mapping = {
        0: '<',
        1: 'v',
        2: '>',
        3: '^'
    }
    print(np.array(list(map(lambda s: mapping[policy(s)] if env.desc.flatten()[s] != b'H' else 'H',
                            np.arange(env.nS)))).reshape(env.ncol, env.nrow))


def greedy_policy(s, gamma=.99):
    return np.argmax([sum([p * (r + gamma * value[next_s])
                          for p, next_s, r, done in env.P[s][a]])
                     for a in range(4)])


def next_v(s, gamma=.99):
    if env.desc.flatten()[s] == b'H':
        return 0.0
    elif env.desc.flatten()[s] == b'G':
        return 0.0
    nexts = max(sum([p * (r + gamma * value[next_s])
                 for p, next_s, r, done in env.P[s][a]])
                for a in range(4))
    return nexts


def value_update(theta=1e-6):
    delta = np.inf
    while delta > theta:
        delta = 0
        for s in value.keys():
            nv = next_v(s)
            delta = max(delta, abs(value[s] - nv))
            value[s] = nv
        # v_pp()


for s, vs in zip(range(env.nS), np.random.normal(scale=1e-2, size=env.nS)):
    value[s] = vs

value_update()

ep_rewards = []
for ep in range(5):
    done = False
    rewards = []
    s = env.reset()
    while not done:
        #env.render()
        action = greedy_policy(s)
        s, reward, done, info = env.step(action)
        rewards.append(reward)
    reward = np.sum(rewards)
    # if reward != 1:
    #     print("Got reward %.2f in %d steps" % (reward, len(rewards)))
    #     break
    # print("Got reward %.2f in %d steps" % (reward, len(rewards)))
    ep_rewards.append(reward)
print("Avg reward: %.2f" % np.mean(ep_rewards))
#env.monitor.close()
np.save("../hw2/fl_optimal_vf", np.array(list(value.values())))
v_pp()
#p_pp(greedy_policy)
print(list(map(greedy_policy, range(63))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning algorithm.')
    parser.add_argument('--env', '-e', type=str, default='FrozenLake8x8-v0', nargs='?',
                        help='The environment to use')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=25000, nargs='?',
                        help='Number of episodes')
    parser.add_argument('--gamma', '-g', metavar='g', type=float, default=0.99, nargs='?',
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