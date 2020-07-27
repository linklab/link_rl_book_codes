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


def main():
    args = argument_parse()
    print_args(args)

    env = gym.make('CartPole-v0')

    dueling_double_dqn_agent = DuelingDoubleDqnAgent(env, args)
    dueling_double_dqn_agent.print_q_network_and_replay_memory_type()
    dueling_double_dqn_agent.learn(args)
    dueling_double_dqn_agent.save_model()

    dueling_double_dqn_agent2 = DuelingDoubleDqnAgent(env, args)
    dueling_double_dqn_agent2.load_model()
    execution(env, dueling_double_dqn_agent2)


if __name__ == "__main__":
    main()