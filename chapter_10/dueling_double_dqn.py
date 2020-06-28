from tensorflow.keras.layers import Add

from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dqn import *
from chapter_10.dueling_dqn import DuelingQNetwork

log_dir = 'logs/dueling_double_dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class DuelingDoubleDqnAgent(DoubleDqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "dueling_double_dqn_agent"
        self.train_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def main():
    env = gym.make('CartPole-v1')
    dueling_double_dqn_agent = DuelingDoubleDqnAgent(env)
    last_episode = dueling_double_dqn_agent.learn()
    print("Learning-completion Episode: {0}".format(last_episode))

    make_video(env, dueling_double_dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dueling_double_dqn/'