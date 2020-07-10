import sys, os
idx = os.getcwd().index("link_rl_book_codes")
PROJECT_HOME = os.path.join(os.getcwd()[:idx-1], "link_rl_book_codes")
sys.path.append(PROJECT_HOME)

from chapter_10.dqn import *
from chapter_10.per_dueling_double_dqn import PerDuelingDoubleDqnAgent
from environments.pong import PongWrappingEnv, PONG_UP_ACTION, PONG_DOWN_ACTION

log_dir = 'logs/cnn_dqn_pong/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class CnnPongQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(CnnPongQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.conv1 = kl.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=state_dim)
        self.conv2 = kl.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.pool1 = kl.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = kl.Dropout(0.25)
        self.flat = kl.Flatten()
        self.dense1 = kl.Dense(units=64, activation='relu')
        self.drop2 = kl.Dropout(0.5)
        self.dense2 = kl.Dense(units=32, activation='relu')

        self.value_output_layer = kl.Dense(units=1, activation='linear')
        self.advantage_output_layer = kl.Dense(units=action_dim, activation='linear')

        self.output_layer = kl.Add()

    def forward(self, state):
        z = self.conv1(state)
        z = self.conv2(z)
        z = self.pool1(z)
        z = self.drop1(z)
        z = self.flat(z)
        z = self.dense1(z)
        z = self.drop2(z)
        z = self.dense2(z)

        value = self.value_output_layer(z)
        advantage = self.advantage_output_layer(z)

        output = self.output_layer(inputs=[value, advantage])
        return output

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            if action == 0:
                return PONG_UP_ACTION
            else:
                return PONG_DOWN_ACTION
        else:
            state = tf.expand_dims(state, axis=0)
            q_value = self.forward(state)
            action = np.argmax(q_value)
            if action == 0:
                return PONG_UP_ACTION
            else:
                return PONG_DOWN_ACTION


class CnnDqnAgent(PerDuelingDoubleDqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "cnn_dqn_agent"
        self.state_dim = (80, 80)
        self.action_dim = 2

        self.train_q_net = CnnPongQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = CnnPongQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def main():
    print_args()

    env = PongWrappingEnv()
    print(env.observation_space)
    print(env.action_space)

    cnn_dqn_agent = CnnDqnAgent(env)
    cnn_dqn_agent.learn()
    cnn_dqn_agent.save_model()

    cnn_dqn_agent2 = CnnDqnAgent(env)
    cnn_dqn_agent2.load_model()
    execution(env, cnn_dqn_agent2)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/advanced_dqn_agent/'