#https://github.com/marload/DeepRL-TensorFlow2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

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
parser.add_argument('--epsilon_min', type=float, default=0.01)
parser.add_argument('--replay_memory_capacity', type=float, default=8192)
parser.add_argument('--max_episodes', type=float, default=1000)
parser.add_argument('--episode_reward_threshold', type=int, default=200)
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


class QNetwork:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(args.learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_value = self.predict(state)[0]
            return np.argmax(q_value)

    def optimize(self, states, targets):
        hist = self.model.fit(states, targets, epochs=1, verbose=0)
        return hist.history['loss'][0]


class DqnAgent:
    def __init__(self, env):
        self.__name__ = "dqn_agent"
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.train_q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayMemory(args.replay_memory_capacity)
        self.last_episode = 0

    def target_update(self):
        self.target_q_net.model.set_weights(
            self.train_q_net.model.get_weights()
        )

    def q_net_optimize(self):
        states, actions, rewards, next_states, dones = self.buffer.get_random_batch(args.batch_size)
        targets = self.train_q_net.predict(states)
        next_q_values = self.target_q_net.predict(next_states).max(axis=1)
        target_q_values = np.where(dones, rewards, rewards + args.gamma * next_q_values)

        targets[range(args.batch_size), actions] = target_q_values

        loss = self.train_q_net.optimize(states, targets)
        return loss

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

            if avg_episode_reward >= args.episode_reward_threshold:
                self.last_episode = ep
                break

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
    done = False
    state = env.reset()
    while not done:
        env.render()
        epsilon = 0.0
        action = agent.train_q_net.get_action(state, epsilon)
        state, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    env = gym.make('CartPole-v1')
    dqn_agent = DqnAgent(env)
    dqn_agent.learn()
    make_video(env, dqn_agent)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dqn/'