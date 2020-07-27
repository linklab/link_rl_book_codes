# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

from chapter_10.dqn import *
from chapter_10.per_double_dqn import PerDoubleDqnAgent
from chapter_10.pong_cnn_dueling_double_dqn import CnnDuelingDoubleDqnAgent, CnnDuelingQNetwork
from environments.pong import PongWrappingEnv


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon_init', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--replay_memory_capacity', type=int, default=250000)
    parser.add_argument('--epsilon_decay_end_step', type=int, default=1000000)
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--average_length_episode_rewards', type=int, default=10)
    parser.add_argument('--train_end_for_average_episode_rewards', type=int, default=15)
    parser.add_argument('--draw_graph_freq', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--train_render', type=bool, default=False)
    args = parser.parse_args()
    return args


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
