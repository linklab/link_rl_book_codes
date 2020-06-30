from chapter_10.dqn import *

log_dir = 'logs/dueling_dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class DuelingQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = kl.InputLayer(input_shape=(state_dim,))
        self.hidden_layer_1 = kl.Dense(32, activation='relu')
        self.hidden_layer_2 = kl.Dense(16, activation='relu')

        self.value_output_layer = kl.Dense(units=1, activation='linear')
        self.advantage_output_layer = kl.Dense(units=action_dim, activation='linear')

        self.output_layer = kl.Add()

    def forward(self, state):
        z = self.input_layer(state)
        z = self.hidden_layer_1(z)
        z = self.hidden_layer_2(z)

        value = self.value_output_layer(z)
        advantage = self.advantage_output_layer(z)

        output = self.output_layer(inputs=[value, advantage])
        return output

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_value = self.forward(state)[0]
            return np.argmax(q_value)


class DuelingDqnAgent(DqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "dueling_dqn_agent"


def main():
    env = gym.make('CartPole-v0')
    dueling_dqn_agent = DuelingDqnAgent(env)
    dueling_dqn_agent.learn()

    make_video(env, dueling_dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dueling_dqn/'