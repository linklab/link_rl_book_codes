import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import gym
import os
import datetime
from gym import wrappers

print(tf.__version__)


class MyModel(tf.keras.Model):
    def __init__(self, num_features, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = kl.InputLayer(input_shape=(num_features,))
        self.hidden_layer_1 = kl.Dense(128, activation='relu')
        self.hidden_layer_2 = kl.Dense(128, activation='relu')
        self.output_layer = kl.Dense(
            num_actions,
            activation='linear'
        )

    def forward(self, inputs):
        z = self.input_layer(inputs)
        z = self.hidden_layer_1(z)
        z = self.hidden_layer_2(z)
        output = self.output_layer(z)
        return output


class Replay_Buffer:
    def __init__(self, max_buffer_size):
        self.transitions = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []
        }
        self.max_buffer_size = max_buffer_size
        self.current_buffer_size = 0

    def get_random_batch(self, batch_size):
        ids = np.random.randint(low=0, high=len(self.transitions['state']), size=batch_size)
        states = np.asarray([self.transitions['state'][i] for i in ids])
        actions = np.asarray([self.transitions['action'][i] for i in ids])
        rewards = np.asarray([self.transitions['reward'][i] for i in ids])
        next_states = np.asarray([self.transitions['next_state'][i] for i in ids])
        dones = np.asarray([self.transitions['done'][i] for i in ids])
        return states, actions, rewards, next_states, dones

    def add_transition(self, transition):
        if self.current_buffer_size >= self.max_buffer_size:
            for key in self.transitions.keys():
                self.transitions[key].pop(0)

            self.current_buffer_size -= 1

        for key, value in transition.items():
            self.transitions[key].append(value)

        self.current_buffer_size += 1


class DQN:
    def __init__(self, num_features, num_actions, gamma, max_buffer_size, train_min_buffer_size, batch_size, learning_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.gamma = gamma
        self.model = MyModel(num_features, num_actions)

        self.max_buffer_size = max_buffer_size
        self.train_min_buffer_size = train_min_buffer_size
        self.replay_buffer = Replay_Buffer(self.max_buffer_size)

    def predict(self, inputs):
        return self.model.forward(np.atleast_2d(inputs.astype('float32')))

    def train(self, target_q_net):
        if self.replay_buffer.current_buffer_size < self.train_min_buffer_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.get_random_batch(self.batch_size)

        with tf.GradientTape() as tape:
            value_next_states = np.max(target_q_net.predict(next_states), axis=1)
            target_q_values = np.where(dones, rewards, rewards + self.gamma * value_next_states)

            current_q_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1
            )
            loss = tf.math.reduce_mean(tf.square(target_q_values - current_q_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_transition(self, transition):
        self.replay_buffer.add_transition(transition)

    def copy_weights(self, train_q_net):
        variables1 = self.model.trainable_variables
        variables2 = train_q_net.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def do_episode(env, train_q_net, target_q_net, epsilon, copy_step):
    score = 0
    iter = 0
    done = False
    state = env.reset()
    losses = list()
    while not done:
        action = train_q_net.get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        score += reward
        if done:
            reward = -200
            env.reset()

        transition = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
        train_q_net.add_transition(transition)
        state = next_state

        loss = train_q_net.train(target_q_net)
        losses.append(loss)
        iter += 1

        if iter % copy_step == 0:
            target_q_net.copy_weights(train_q_net)

    return score, np.mean(losses)


def make_video(env, train_q_net):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        action = train_q_net.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_step = 25
    num_features = len(env.observation_space.sample())
    num_actions = env.action_space.n
    max_buffer_size = 10000
    min_buffer_size = 100
    batch_size = 2
    learning_rate = 0.01
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    train_q_net = DQN(num_features, num_actions, gamma, max_buffer_size, min_buffer_size, batch_size, learning_rate)
    target_q_net = DQN(num_features, num_actions, gamma, max_buffer_size, min_buffer_size, batch_size, learning_rate)
    target_q_net.copy_weights(train_q_net)

    max_episodes = 50000
    score_list = np.empty(max_episodes)
    epsilon = 0.5
    epsilon_decay = 0.999
    min_epsilon = 0.1
    avg_score = 0

    for episode in range(max_episodes):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # score: cummulative rewards at episode
        score, losses = do_episode(env, train_q_net, target_q_net, epsilon, copy_step)
        score_list[episode] = score
        avg_score = score_list[max(0, episode - 10):(episode + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar('score', score, step=episode)
            tf.summary.scalar('avg score', avg_score, step=episode)
            tf.summary.scalar('avg loss', losses, step=episode)

        if episode % 10 == 0:
            print("episode: {0}, score: {1}, avg score (last 10 episodes): {2:.1f}, epsilon: {3:.3f}, episode loss: {4:.3f}".format(
                episode, score, avg_score, epsilon, losses
            ))
        if avg_score >= 200:
            break

    print("avg reward for last 10 episodes:", avg_score)
    make_video(env, train_q_net)
    env.close()


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dqn/'