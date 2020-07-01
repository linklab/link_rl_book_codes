from chapter_10.dqn import *
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent

log_dir = 'logs/per_dueling_double_dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.nodes = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.nodes[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.nodes):
            return idx
        if s <= self.nodes[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.nodes[left])

    def total(self):
        return self.nodes[0]

    def add(self, data, priority):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.nodes[idx]
        self.nodes[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        priority = self.nodes[idx]
        return idx, self.data[data_idx], priority


class PrioritizedExperienceMemory:  # stored as ( s, a, r, s', p ) in SumTree
    e = 0.001
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.capacity = capacity
        self.sum_tree = SumTree(capacity)

    def reset(self):
        self.sum_tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.alpha

    def put(self, transition, error=10000.0):
        priority = self._getPriority(error)
        self.sum_tree.add(transition, priority)

    def size(self):
        return self.sum_tree.n_entries

    def get_random_batch(self, size):
        priorities = []
        batch = []
        idxs = []

        num_segment = self.sum_tree.total() / size

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(size):
            a = num_segment * i
            b = num_segment * (i + 1)

            s = random.uniform(a, b)
            idx, transition, priority = self.sum_tree.get(s)
            priorities.append(priority)
            batch.append(transition)
            idxs.append(idx)

        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self.sum_tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priority(self, idx, error):
        priority = self._getPriority(error)
        self.sum_tree.update(idx, priority)


class PerDuelingDoubleDqnAgent(DuelingDoubleDqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "per_dueling_double_dqn_agent"
        self.buffer = PrioritizedExperienceMemory(args.replay_memory_capacity)

    def q_net_optimize(self):
        batch, idxs, is_weight = self.buffer.get_random_batch(args.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))

        with tf.GradientTape() as tape:
            selected_actions = np.argmax(self.train_q_net.forward(next_states), axis=1)
            next_q_values = tf.math.reduce_sum(
                self.target_q_net.forward(next_states) * tf.one_hot(selected_actions, self.action_dim),axis=1
            )
            target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)
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


def prioritized_experience_memory_test():
    memory = PrioritizedExperienceMemory(capacity=4)
    memory.put(transition="A")
    memory.put(transition="B", error=2.0)
    memory.put(transition="C", error=3.0)
    memory.put(transition="D", error=100.0)
    c = memory.get_random_batch(10)
    print(memory)
    print(c)


def main():
    env = gym.make('CartPole-v0')
    per_dueling_double_dqn_agent = PerDuelingDoubleDqnAgent(env)
    per_dueling_double_dqn_agent.learn()

    make_video(env, per_dueling_double_dqn_agent)


if __name__ == "__main__":
    #prioritized_experience_memory_test()
    main()
    # tensorboard --logdir 'logs/double_dqn/'