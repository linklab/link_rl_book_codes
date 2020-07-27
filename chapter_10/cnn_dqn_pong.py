# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

from chapter_10.double_dqn import DoubleDqnAgent
from chapter_10.dqn import *
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
    parser.add_argument('--max_steps', type=int, default=5000000)
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--draw_graph_freq', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--train_render', type=bool, default=False)
    args = parser.parse_args()
    return args


class CnnDuelingQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(CnnDuelingQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.conv1 = kl.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dim)
        self.conv2 = kl.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.conv3 = kl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.flat = kl.Flatten()
        self.dense1 = kl.Dense(units=512, activation='relu')

        self.value_output_layer = kl.Dense(units=1, activation='linear')
        self.advantage_output_layer = kl.Dense(units=action_dim, activation='linear')

        self.output_layer = kl.Add()

        self.num_actions_executed = {}
        self.reset_num_actions_executed()

    def reset_num_actions_executed(self):
        for action in range(self.action_dim):
            self.num_actions_executed[action] = 0

    def call(self, state, **kwargs):
        return self.forward(state)

    def forward(self, state):
        z = self.conv1(state)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.flat(z)
        z = self.dense1(z)

        value = self.value_output_layer(z)
        advantage = self.advantage_output_layer(z)
        output = self.output_layer(inputs=[value, advantage])

        return output

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state = tf.expand_dims(state, axis=0)
            q_value = self.forward(state)
            action = np.argmax(q_value)

        self.num_actions_executed[action] += 1
        return action


class CnnDuelingDoubleDqnAgent(DoubleDqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "cnn_dueling_double_dqn"
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.train_q_net = CnnDuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = CnnDuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def main():
    args = argument_parse()
    print_args(args)

    env = PongWrappingEnv()

    print(env.observation_space)
    print(env.action_space)

    cnn_dueling_double_dqn_agent = CnnDuelingDoubleDqnAgent(env, args)
    cnn_dueling_double_dqn_agent.print_q_network_and_replay_memory_type()
    cnn_dueling_double_dqn_agent.learn(args)
    cnn_dueling_double_dqn_agent.save_model()

    cnn_dueling_double_dqn_agent2 = CnnDuelingDoubleDqnAgent(env, args)
    cnn_dueling_double_dqn_agent2.load_model()
    execution(env, cnn_dueling_double_dqn_agent2)


if __name__ == "__main__":
    main()