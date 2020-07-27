from chapter_10.dqn import *


class DoubleDqnAgent(DqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "double_dqn"

    def q_net_optimize(self, args):
        states, actions, rewards, next_states, dones = self.buffer.get_random_batch(args.batch_size)

        selected_actions = np.argmax(self.train_q_net.forward(next_states), axis=1)
        next_q_values = tf.math.reduce_sum(
            self.target_q_net.forward(next_states) * tf.one_hot(selected_actions, self.action_dim), axis=1
        )
        target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)

        with tf.GradientTape() as tape:
            current_q_values = tf.math.reduce_sum(
                self.train_q_net.forward(states) * tf.one_hot(actions, self.action_dim), axis=1
            )
            loss = tf.math.reduce_mean(tf.square(target_q_values - current_q_values))

        variables = self.train_q_net.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()


def main():
    args = argument_parse()
    print_args(args)

    env = gym.make('CartPole-v0')

    double_dqn_agent = DoubleDqnAgent(env, args)
    double_dqn_agent.print_q_network_and_replay_memory_type()
    double_dqn_agent.learn(args)
    double_dqn_agent.save_model()

    double_dqn_agent2 = DoubleDqnAgent(env, args)
    double_dqn_agent2.load_model()
    execution(env, double_dqn_agent2)


if __name__ == "__main__":
    main()
