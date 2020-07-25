from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dqn import *
from chapter_10.dueling_dqn import DuelingQNetwork
from environments.pong import PongWrappingEnv


class DuelingDoubleDqnAgent(DoubleDqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "dueling_double_dqn_agent"
        self.train_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def main():
    env = gym.make('CartPole-v0')

    dueling_double_dqn_agent = DuelingDoubleDqnAgent(env)
    dueling_double_dqn_agent.print_q_network()
    dueling_double_dqn_agent.learn()
    dueling_double_dqn_agent.save_model()

    dueling_double_dqn_agent2 = DuelingDoubleDqnAgent(env)
    dueling_double_dqn_agent2.load_model()
    execution(env, dueling_double_dqn_agent2)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dueling_double_dqn/'