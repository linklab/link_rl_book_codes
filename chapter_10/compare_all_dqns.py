import gym
import os

from chapter_10.dqn import DqnAgent, args
from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent
from chapter_10.dueling_dqn import DuelingDqnAgent
from chapter_10.per_dueling_double_dqn import PerDuelingDoubleDqnAgent

from matplotlib import pyplot as plt
import numpy as np

max_runs = 100

dqn_variants = [
    'dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn', 'per_dueling_double_dqn'
]

def main():
    performance = np.zeros(shape=(len(dqn_variants), args.max_episodes))

    for run in range(max_runs):
        print("######### run:{0} #########".format(run))
        env = gym.make('CartPole-v0')
        dqn_agent = DqnAgent(env)
        dqn_agent.learn()

        env = gym.make('CartPole-v0')
        double_dqn_agent = DoubleDqnAgent(env)
        double_dqn_agent.learn()

        env = gym.make('CartPole-v0')
        dueling_dqn_agent = DuelingDqnAgent(env)
        dueling_dqn_agent.learn()

        env = gym.make('CartPole-v0')
        dueling_double_dqn_agent = DuelingDoubleDqnAgent(env)
        dueling_double_dqn_agent.learn()

        env = gym.make('CartPole-v0')
        per_dueling_double_dqn_agent = PerDuelingDoubleDqnAgent(env)
        per_dueling_double_dqn_agent.learn()

        performance[0, :] += dqn_agent.episode_reward_list
        performance[1, :] += double_dqn_agent.episode_reward_list
        performance[2, :] += dueling_dqn_agent.episode_reward_list
        performance[3, :] += dueling_double_dqn_agent.episode_reward_list
        performance[4, :] += per_dueling_double_dqn_agent.episode_reward_list

        avg_performance = performance / (run + 1)

        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['', 's', '^', 'o', 'x', '']
        plt.figure(figsize=(12, 5))
        for i in range(len(dqn_variants)):
            plt.plot(
                range(args.max_episodes), avg_performance[i],
                label=dqn_variants[i],
                linestyle=line_styles[i], marker=markers[i]
            )

        plt.xlabel('Episodes')
        plt.ylabel('Episode rewards')
        plt.title('Performance Comparison (run: {0})'.format(run + 1))
        plt.legend()

        plt.savefig("images/comparison_all_dqns.png")
        plt.close()


if __name__ == "__main__":
    if not os.path.exists('images/'):
        os.makedirs('images/')
    main()