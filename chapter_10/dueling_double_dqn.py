from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dqn import *
from chapter_10.dueling_dqn import DuelingQNetwork


class DuelingDoubleDqnAgent(DoubleDqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "dueling_double_dqn"
        self.train_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def train(args):
    env = gym.make(args.env)

    dueling_double_dqn_agent = DuelingDoubleDqnAgent(env, args)
    dueling_double_dqn_agent.print_q_network_and_replay_memory_type()
    dueling_double_dqn_agent.learn()
    dueling_double_dqn_agent.save_model()


def play(args):
    env = gym.make(args.env)
    dueling_double_dqn_agent2 = DuelingDoubleDqnAgent(env, args)
    dueling_double_dqn_agent2.load_model()
    execution(env, dueling_double_dqn_agent2)


def main():
    args = argument_parse()
    print_args(args)

    train(args)

    # 테스트시에는 CartPole-v1을 사용하여 테스트
    # CartPole-v1의 MAX 스텝: 500 vs. CartPole-v0의 MAX 스텝: 200
    args.env = 'CartPole-v1'
    play(args)


if __name__ == "__main__":
    main()