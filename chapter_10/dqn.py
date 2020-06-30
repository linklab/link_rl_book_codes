#https://github.com/marload/DeepRL-TensorFlow2
import tensorflow as tf
import tensorflow.keras.layers as kl

import datetime
import gym
import argparse
import numpy as np
from collections import deque
import random
from gym import wrappers
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_decay', type=float, default=0.999)
parser.add_argument('--epsilon_min', type=float, default=0.001)
parser.add_argument('--replay_memory_capacity', type=float, default=8192)
parser.add_argument('--max_episodes', type=float, default=75)
args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

log_dir = 'logs/dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def put(self, transition):
        self.buffer.append(transition)

    def get_random_batch(self, size):
        batch = random.sample(self.buffer, size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*batch))
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = kl.InputLayer(input_shape=(state_dim,))
        self.hidden_layer_1 = kl.Dense(units=32, activation='relu')
        self.hidden_layer_2 = kl.Dense(units=16, activation='relu')
        self.output_layer = kl.Dense(units=action_dim, activation='linear')

    def forward(self, state):
        z = self.input_layer(state)
        z = self.hidden_layer_1(z)
        z = self.hidden_layer_2(z)
        output = self.output_layer(z)
        return output

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_value = self.forward(state)[0]
            return np.argmax(q_value)


class DqnAgent:
    def __init__(self, env):
        self.__name__ = "dqn_agent"
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.optimizer = tf.optimizers.Adam(args.learning_rate)
        self.train_q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayMemory(args.replay_memory_capacity)
        self.episode_reward_list = []

    def target_update(self):
        train_q_net_variables = self.train_q_net.trainable_variables
        target_q_net_variables = self.target_q_net.trainable_variables
        for v1, v2 in zip(train_q_net_variables, target_q_net_variables):
            v2.assign(v1.numpy())

    def q_net_optimize(self):
        states, actions, rewards, next_states, dones = self.buffer.get_random_batch(args.batch_size)

        with tf.GradientTape() as tape:
            next_q_values = np.max(self.target_q_net.forward(next_states), axis=1)
            target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)
            current_q_values = tf.math.reduce_sum(
                self.train_q_net.forward(states) * tf.one_hot(actions, self.action_dim), axis=1
            )

            loss = tf.math.reduce_mean(tf.square(target_q_values - current_q_values))

        variables = self.train_q_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()

    def learn(self):
        episode_rewards_last_10 = deque(maxlen=10)
        epsilon = args.epsilon

        for ep in range(args.max_episodes):
            state = self.env.reset()

            episode_reward = 0
            episode_loss = 0.0
            done = False

            while not done:
                epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
                action = self.train_q_net.get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                transition = [state, action, reward * 0.01, next_state, done]
                self.buffer.put(transition)
                episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    episode_loss += self.q_net_optimize()

                state = next_state

            episode_rewards_last_10.append(episode_reward)
            avg_episode_reward = np.array(episode_rewards_last_10).mean()

            self.target_update()

            self.write_performance(ep, epsilon, episode_reward, avg_episode_reward, episode_loss)
            self.episode_reward_list.append(avg_episode_reward)

    def write_performance(self, ep, epsilon, episode_reward, avg_episode_reward, episode_loss):
        print(
            "[{0}] Episode: {1}(Epsilon: {2:.3f}), Episode reward: {3}, Average episode reward (last 10 episodes): {4:.3f}, Episode loss: {5:.5f}".format(
                self.__name__, ep, epsilon, episode_reward, avg_episode_reward, episode_loss
            ))

        with summary_writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward, step=ep)
            tf.summary.scalar('Episode Reward (average last 10 episodes)', avg_episode_reward, step=ep)
            tf.summary.scalar('Episode Loss', episode_loss, step=ep)


def make_video(env, agent):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    epsilon = 0.0

    done = False
    state = env.reset()
    while not done:
        env.render()
        action = agent.train_q_net.get_action(state, epsilon)
        state, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    env = gym.make('CartPole-v0')
    dqn_agent = DqnAgent(env)
    last_episode = dqn_agent.learn()
    print("Learning-completion Episode: {0}".format(last_episode))

    make_video(env, dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dqn/'