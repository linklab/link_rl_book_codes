from chapter_10.dqn import *
from environments.pong import PongWrappingEnv


class DoubleDqnAgent(DqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "double_dqn_agent"

    def q_net_optimize(self):
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
    env = gym.make('CartPole-v0')

    double_dqn_agent = DoubleDqnAgent(env)
    double_dqn_agent.learn()
    double_dqn_agent.save_model()

    double_dqn_agent2 = DoubleDqnAgent(env)
    double_dqn_agent2.load_model()
    execution(env, double_dqn_agent2)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/double_dqn/'