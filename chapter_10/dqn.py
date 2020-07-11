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
parser.add_argument('--replay_memory_capacity', type=int, default=8192)
parser.add_argument('--max_episodes', type=int, default=100)
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

log_dir = 'logs/dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


def print_args():
    print("##############################################")
    print("gamma: {0}".format(args.gamma))
    print("learning_rate: {0}".format(args.learning_rate))
    print("batch_size: {0}".format(args.batch_size))
    print("epsilon: {0}".format(args.epsilon))
    print("epsilon_decay: {0}".format(args.epsilon_decay))
    print("epsilon_min: {0}".format(args.epsilon_min))
    print("replay_memory_capacity: {0}".format(args.replay_memory_capacity))
    print("max_episodes: {0}".format(args.max_episodes))
    print("verbose: {0}".format(args.verbose))
    print("##############################################")


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

        self.num_action_executed = {}
        for action in range(action_dim):
            self.num_action_executed[action] = 0

    def forward(self, state):
        z = self.input_layer(state)
        z = self.hidden_layer_1(z)
        z = self.hidden_layer_2(z)
        output = self.output_layer(z)
        return output

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            self.num_action_executed[action] += 1
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_value = self.forward(state)[0]
            action = int(np.argmax(q_value))
            self.num_action_executed[action] += 1
        return action


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

        if not os.path.exists(os.path.join(os.getcwd(), 'models')):
            os.makedirs(os.path.join(os.getcwd(), 'models'))

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
                self.env.render()
                epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
                action = self.train_q_net.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                if args.verbose:
                    print("State: {0}, Action: {1}, Next State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
                        state.shape, action, next_state.shape, reward, done, info
                    ))

                transition = [state, action, reward * 0.01, next_state, done]
                self.buffer.put(transition)
                episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    episode_loss += self.q_net_optimize()

                state = next_state
            print()

            episode_rewards_last_10.append(episode_reward)
            avg_episode_reward = np.array(episode_rewards_last_10).mean()

            self.target_update()

            self.write_performance(ep, epsilon, episode_reward, avg_episode_reward, episode_loss)
            self.episode_reward_list.append(avg_episode_reward)

    def save_model(self):
        self.train_q_net.save_weights(
            os.path.join(os.getcwd(), 'models', 'dqn_{0}.tf'.format(self.__name__)), save_format="tf"
        )

    def load_model(self):
        self.train_q_net.load_weights(
            os.path.join(os.getcwd(), 'models', 'dqn_{0}.tf'.format(self.__name__))
        )

    def write_performance(self, ep, epsilon, episode_reward, avg_episode_reward, episode_loss):
        str_info = "[{0}] Episode: {1}(Epsilon: {2:.3f}), Episode reward: {3}, " \
                   "Average episode reward (last 10 episodes): {4:.3f}, Episode loss: {5:.5f}, Buffer Size: {6}".format(
            self.__name__, ep, epsilon, episode_reward, avg_episode_reward, episode_loss, self.buffer.size()
        )

        str_info += "Number of actions: "
        for action in self.train_q_net.num_action_executed:
            str_info += "[{0}: {1}] ".format(action, self.train_q_net.num_action_executed[action])
        print(str_info)

        with summary_writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward, step=ep)
            tf.summary.scalar('Episode Reward (average last 10 episodes)', avg_episode_reward, step=ep)
            tf.summary.scalar('Episode Loss', episode_loss, step=ep)


def execution(env, agent, make_video=False):
    if make_video:
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
    print_args()

    env = gym.make('CartPole-v0')
    dqn_agent = DqnAgent(env)
    dqn_agent.learn()
    dqn_agent.save_model()

    dqn_agent2 = DqnAgent(env)
    dqn_agent2.load_model()
    execution(env, dqn_agent2)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/dqn/'