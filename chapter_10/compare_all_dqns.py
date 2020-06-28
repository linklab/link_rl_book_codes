import gym
import os

from chapter_10.dqn import DqnAgent
from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent
from chapter_10.dueling_dqn import DuelingDqnAgent
from chapter_10.per_dueling_double_dqn import PerDuelingDoubleDqnAgent

from matplotlib import pyplot as plt
import numpy as np

max_runs = 50

dqn_variants = [
    'dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn', 'per_dueling_double_dqn'
]

def main():
    sum_performance = np.zeros(shape=(len(dqn_variants), ))
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

        sum_performance[0] += dqn_last_episode
        sum_performance[1] += double_dqn_last_episode
        sum_performance[2] += dueling_dqn_last_episode
        sum_performance[3] += dueling_double_dqn_last_episode
        sum_performance[4] += per_dueling_double_dqn_last_episode

        mean_performance = sum_performance / (run + 1)

        plt.figure(figsize=(12, 5))
        plt.bar(dqn_variants, mean_performance)
        plt.xlabel('DQN Variants')
        plt.ylabel('Learning-completion Episode')
        plt.title('Performance Comparison (run: {0})'.format(run + 1))
        plt.savefig("images/comparison_all_dqns.png")
        plt.close()


if __name__ == "__main__":
    if not os.path.exists('images/'):
        os.makedirs('images/')
    main()