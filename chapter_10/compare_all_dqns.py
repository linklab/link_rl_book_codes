import gym
import os

from chapter_10.dqn import DqnAgent
from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent
from chapter_10.dueling_dqn import DuelingDqnAgent
from chapter_10.per_dueling_double_dqn import PerDuelingDoubleDqnAgent

from matplotlib import pyplot as plt
import numpy as np

max_runs = 30

dqn_variants = [
    'dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn', 'per_dueling_double_dqn'
]

def main():
    performance = np.zeros((len(dqn_variants), max_runs))
    for run in range(max_runs):
        print("######### run:{0} #########".format(run))
        env = gym.make('CartPole-v1')
        dqn_agent = DqnAgent(env)
        dqn_last_episode = dqn_agent.learn()

        env = gym.make('CartPole-v1')
        double_dqn_agent = DoubleDqnAgent(env)
        double_dqn_last_episode = double_dqn_agent.learn()

        env = gym.make('CartPole-v1')
        dueling_dqn_agent = DuelingDqnAgent(env)
        dueling_dqn_last_episode = dueling_dqn_agent.learn()

        env = gym.make('CartPole-v1')
        dueling_double_dqn_agent = DuelingDoubleDqnAgent(env)
        dueling_double_dqn_last_episode = dueling_double_dqn_agent.learn()

        env = gym.make('CartPole-v1')
        per_dueling_double_dqn_agent = PerDuelingDoubleDqnAgent(env)
        per_dueling_double_dqn_last_episode = per_dueling_double_dqn_agent.learn()

        performance[0, run] = dqn_last_episode
        performance[1, run] = double_dqn_last_episode
        performance[2, run] = dueling_dqn_last_episode
        performance[3, run] = dueling_double_dqn_last_episode
        performance[4, run] = per_dueling_double_dqn_last_episode

        mean_performance = performance.mean(axis=1)

        plt.figure(figsize=(12, 3))
        plt.bar(dqn_variants, mean_performance)
        plt.xlabel('DQN Variants')
        plt.ylabel('Learning-completion Episode')
        plt.title('DQN Variants (run: {0})'.format(run))
        plt.savefig("images/comparison_all_dqns.png")
        plt.close()


if __name__ == "__main__":
    if not os.path.exists('images/'):
        os.makedirs('images/')
    main()