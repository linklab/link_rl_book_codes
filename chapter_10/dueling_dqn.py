from chapter_10.dqn import *


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

        self.num_actions_executed = {}
        self.reset_num_actions_executed()

    def reset_num_actions_executed(self):
        for action in range(self.action_dim):
            self.num_actions_executed[action] = 0

    def call(self, state, **kwargs):
        return self.forward(state)

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
            action = random.randint(0, self.action_dim - 1)
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_value = self.forward(state)[0]
            action = int(np.argmax(q_value))

        self.num_actions_executed[action] += 1
        return action


class DuelingDqnAgent(DqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "dueling_dqn"
        self.train_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_q_net = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_update()


def train(args):
    env = gym.make(args.env)

    dueling_dqn_agent = DuelingDqnAgent(env, args)
    dueling_dqn_agent.print_q_network_and_replay_memory_type()
    dueling_dqn_agent.learn()
    dueling_dqn_agent.save_model()


def play(args):
    env = gym.make(args.env)

    dueling_dqn_agent2 = DuelingDqnAgent(env, args)
    dueling_dqn_agent2.load_model()
    execution(env, dueling_dqn_agent2)


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
