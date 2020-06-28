from chapter_10.dqn import *

log_dir = 'logs/double_dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class DoubleDqnAgent(DqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "double_dqn_agent"

    def q_net_optimize(self):
        states, actions, rewards, next_states, dones = self.buffer.get_random_batch(args.batch_size)
        targets = self.train_q_net.predict(states)
        next_q_values = self.target_q_net.predict(next_states)[
            range(args.batch_size), np.argmax(self.train_q_net.predict(next_states), axis=1)
        ]
        target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)

        targets[range(args.batch_size), actions] = target_q_values

        loss = self.train_q_net.optimize(states, targets)
        return loss


def main():
    env = gym.make('CartPole-v1')
    double_dqn_agent = DoubleDqnAgent(env)
    double_dqn_agent.learn()
    make_video(env, double_dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/double_dqn/'