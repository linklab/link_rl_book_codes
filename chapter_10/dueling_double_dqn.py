from tensorflow.keras.layers import Add

from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dqn import *

log_dir = 'logs/dueling_double_dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class DuelingQNetwork(QNetwork):
    def create_model(self):
        state_input = Input((self.state_dim,))
        backbone_1 = Dense(32, activation='relu')(state_input)
        backbone_2 = Dense(16, activation='relu')(backbone_1)

        value_output = Dense(1)(backbone_2)
        advantage_output = Dense(self.action_dim)(backbone_2)

        output = Add()([value_output, advantage_output])

        model = tf.keras.Model(state_input, output)
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(args.learning_rate))
        return model


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
    dueling_double_dqn_agent.learn()
    make_video(env, dueling_double_dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/double_dqn/'