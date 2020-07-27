from chapter_10.dqn import *
from chapter_10.per_dqn import PerDqnAgent


class PerDoubleDqnAgent(PerDqnAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__name__ = "per_double_dqn"

    def q_net_optimize(self, args):
        batch, idxs, is_weight = self.buffer.get_random_batch(args.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))

        # Double DQN
        selected_actions = np.argmax(self.train_q_net.forward(next_states), axis=1)
        next_q_values = tf.math.reduce_sum(
            self.target_q_net.forward(next_states) * tf.one_hot(selected_actions, self.action_dim), axis=1
        )
        target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)

        with tf.GradientTape() as tape:
            current_q_values = tf.math.reduce_sum(
                self.train_q_net.forward(states) * tf.one_hot(actions, self.action_dim), axis=1
            )
            loss = tf.math.reduce_mean(tf.square(target_q_values - current_q_values) * is_weight)

        # td_error로 우선순위 업데이트
        td_error = np.abs(target_q_values - current_q_values)
        for i in range(args.batch_size):
            idx = idxs[i]
            self.buffer.update_priority(idx, td_error[i])

        # train_q_net 가중치 갱신
        variables = self.train_q_net.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()


def main():
    args = argument_parse()
    print_args(args)

    env = gym.make('CartPole-v0')

    per_double_dqn_agent = PerDoubleDqnAgent(env, args)
    per_double_dqn_agent.print_q_network_and_replay_memory_type()
    per_double_dqn_agent.learn(args)
    per_double_dqn_agent.save_model()

    per_double_dqn_agent2 = PerDoubleDqnAgent(env, args)
    per_double_dqn_agent2.save_model()
    execution(env, per_double_dqn_agent2)


if __name__ == "__main__":
    main()