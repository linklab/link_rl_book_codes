from chapter_10.dqn import *
from chapter_10.dueling_dqn import DuelingQNetwork
from chapter_10.per_dqn import PerDqnAgent


class PerDuelingDqnAgent(PerDqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "per_dueling_dqn"
        self.train_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def train(args):
    env = gym.make(args.env)

    per_dueling_dqn_agent = PerDuelingDqnAgent(env, args)
    per_dueling_dqn_agent.print_q_network_and_replay_memory_type()
    per_dueling_dqn_agent.learn()
    per_dueling_dqn_agent.save_model()


def play(args):
    env = gym.make(args.env)

    per_dueling_dqn_agent2 = PerDuelingDqnAgent(env, args)
    per_dueling_dqn_agent2.save_model()
    execution(env, per_dueling_dqn_agent2)


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