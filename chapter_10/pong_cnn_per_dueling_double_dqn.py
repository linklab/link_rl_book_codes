# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

from chapter_10.dqn import *
from chapter_10.per_double_dqn import PerDoubleDqnAgent
from chapter_10.pong_cnn_dueling_double_dqn import CnnDuelingDoubleDqnAgent, CnnDuelingQNetwork
from environments.pong import PongWrappingEnv


class CnnPerDuelingDoubleDqnAgent(PerDoubleDqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "cnn_per_dueling_double_dqn"
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.train_q_net = CnnDuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = CnnDuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def train(args):
    env = PongWrappingEnv()
    print(env.observation_space)
    print(env.action_space)

    cnn_per_dueling_double_dqn_agent = CnnPerDuelingDoubleDqnAgent(env, args)
    cnn_per_dueling_double_dqn_agent.print_q_network_and_replay_memory_type()
    cnn_per_dueling_double_dqn_agent.learn()
    cnn_per_dueling_double_dqn_agent.save_model()


def play(args):
    env = PongWrappingEnv()

    cnn_per_dueling_double_dqn_agent2 = CnnDuelingDoubleDqnAgent(env, args)
    cnn_per_dueling_double_dqn_agent2.load_model()
    execution(env, cnn_per_dueling_double_dqn_agent2)


def main():
    args = argument_parse()
    print_args(args)

    train(args)

    play(args)


if __name__ == "__main__":
    main()
