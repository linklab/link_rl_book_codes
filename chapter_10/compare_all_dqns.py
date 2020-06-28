import gym
import os

from chapter_10.dqn import DqnAgent, args
from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent
from chapter_10.per_dueling_double_dqn import PerDuelingDoubleDqnAgent

from matplotlib import pyplot as plt
import numpy as np

max_runs = 10

def main():
    performance = np.zeros((4, max_runs))
    for run in range(max_runs):
        print("######### run:{0} #########".format(run))
        env = gym.make('CartPole-v1')
        dqn_agent = DqnAgent(env)
        dqn_agent.learn()

        env = gym.make('CartPole-v1')
        double_dqn_agent = DoubleDqnAgent(env)
        double_dqn_agent.learn()

        env = gym.make('CartPole-v1')
        dueling_double_dqn_agent = DuelingDoubleDqnAgent(env)
        dueling_double_dqn_agent.learn()

        env = gym.make('CartPole-v1')
        per_dueling_double_dqn_agent = PerDuelingDoubleDqnAgent(env)
        per_dueling_double_dqn_agent.learn()

        performance[0, run] = dqn_agent.last_episode
        performance[1, run] = double_dqn_agent.last_episode
        performance[2, run] = dueling_double_dqn_agent.last_episode
        performance[3, run] = per_dueling_double_dqn_agent.last_episode

        mean_performance = performance.mean(axis=1)

        plt.bar(
            ['dqn', 'double+', 'dueling++', 'per+++'],
            mean_performance
        )
        plt.xlabel('DQN Variants')
        plt.ylabel('Episode Reward')

        plt.savefig("images/comparison_all_dqns.png")
        plt.close()


if __name__ == "__main__":
    if not os.path.exists('images/'):
        os.makedirs('images/')
    main()