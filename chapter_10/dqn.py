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

from visdom import Visdom

vis = Visdom()
episode_reward_plt = vis.line(X=[0], Y=[0], opts=dict(title='Episode Reward', showlegend=False))
avg_episode_reward_plt = vis.line(X=[0], Y=[0], opts=dict(title='Average Episode Reward', showlegend=False))
episode_loss_plt = vis.line(X=[0], Y=[0], opts=dict(title='Episode Loss', showlegend=False))

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epsilon_init', type=float, default=1.0)
parser.add_argument('--epsilon_min', type=float, default=0.01)
parser.add_argument('--replay_memory_capacity', type=int, default=250000)
parser.add_argument('--epsilon_decay_end_step', type=int, default=1000000)
parser.add_argument('--max_steps', type=int, default=5000000)
parser.add_argument('--target_net_update_freq', type=int, default=1000)
parser.add_argument('--draw_graph_freq', type=int, default=10)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--train_render', type=bool, default=False)
args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def print_args():
    print("##############################################")
    print("gamma: {0}".format(args.gamma))
    print("learning_rate: {0}".format(args.learning_rate))
    print("batch_size: {0}".format(args.batch_size))
    print("epsilon_init: {0}".format(args.epsilon_init))
    print("epsilon_min: {0}".format(args.epsilon_min))
    print("epsilon_decay_end_step: {0}".format(args.epsilon_decay_end_step))
    print("max_steps: {0}".format(args.max_steps))
    print("replay_memory_capacity: {0}".format(args.replay_memory_capacity))
    print("target_net_update_freq: {0}".format(args.target_net_update_freq))
    print("draw_graph_freq: {0}".format(args.draw_graph_freq))
    print("verbose: {0}".format(args.verbose))
    print("train_render: {0}".format(args.train_render))
    print("##############################################")


def linear_interpolation(start_step, end_step, current_step, final_p, initial_p):
    schedule_timesteps = end_step - start_step
    step_offset = current_step - start_step
    fraction = min(float(step_offset) / schedule_timesteps, 1)
    return min(initial_p + fraction * (final_p - initial_p), initial_p)


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
        output = self.output_layer(z)
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

    def print_q_network(self):
        if type(self.state_dim) is int:
            one_input_shape = tuple([1] + [self.state_dim,])
        else:
            one_input_shape = tuple([1] + list(self.state_dim))
        print("one input shape: {0}".format(one_input_shape))
        self.train_q_net.build(input_shape=one_input_shape)
        self.train_q_net.summary()

    def target_update(self):
        train_q_net_variables = self.train_q_net.trainable_variables
        target_q_net_variables = self.target_q_net.trainable_variables
        for v1, v2 in zip(train_q_net_variables, target_q_net_variables):
            v2.assign(v1.numpy())

    def q_net_optimize(self):
        states, actions, rewards, next_states, dones = self.buffer.get_random_batch(args.batch_size)

        next_q_values = np.where(dones, 0, np.max(self.target_q_net.forward(next_states), axis=1))
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

    def learn(self):
        episode_rewards_last_10 = deque(maxlen=10)
        epsilon = args.epsilon_init

        total_steps = 0
        episode = 0
        train_done = False

        while not train_done:
            episode += 1
            # new episode started
            state = self.env.reset()

            episode_steps = 0
            episode_reward = 0
            episode_loss = 0.0
            done = False

            while not done:
                total_steps += 1
                episode_steps += 1

                if args.train_render:
                    self.env.render()

                epsilon = linear_interpolation(
                    start_step=0,
                    end_step=args.epsilon_decay_end_step,
                    current_step=total_steps,
                    final_p=args.epsilon_min,
                    initial_p=args.epsilon_init
                )

                action = self.train_q_net.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                if args.verbose:
                    print("State: {0}, Action: {1}, Next State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
                        state.shape,
                        action,
                        next_state.shape if next_state is not None else "None",
                        reward,
                        done,
                        info
                    ))

                transition = [state, action, reward * 0.01, next_state, done]
                self.buffer.put(transition)

                episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    episode_loss += self.q_net_optimize()

                if total_steps % args.target_net_update_freq == 0:
                    self.target_update()

                state = next_state

                if total_steps >= args.max_steps:
                    train_done = True
                    break

            episode_rewards_last_10.append(episode_reward)
            avg_episode_reward = np.array(episode_rewards_last_10).mean()

            self.write_performance(
                episode, epsilon, episode_reward, avg_episode_reward, episode_loss, total_steps, episode_steps
            )
            self.episode_reward_list.append(avg_episode_reward)
            self.train_q_net.reset_num_actions_executed()

            if args.verbose:
                print()

    def save_model(self):
        self.train_q_net.save_weights(
            os.path.join(os.getcwd(), 'models', 'dqn_{0}.tf'.format(self.__name__)), save_format="tf"
        )

    def load_model(self):
        self.train_q_net.load_weights(
            os.path.join(os.getcwd(), 'models', 'dqn_{0}.tf'.format(self.__name__))
        )

    def write_performance(self, episode, epsilon, episode_reward, avg_episode_reward, episode_loss, total_steps, episode_steps):
        str_info = "[{0}] Episode: {1}, Eps.: {2:.3f}, Episode reward: {3}, Avg. episode reward (last 10): {4:.3f}, " \
                   "Episode loss: {5:.5f}, Buffer size: {6}, Total steps: {7} ({8})".format(
            self.__name__, episode, epsilon, episode_reward, avg_episode_reward,
            episode_loss, self.buffer.size(), total_steps, episode_steps
        )

        str_info += ", Number of actions: "
        for action in self.train_q_net.num_actions_executed:
            str_info += "[{0}: {1}] ".format(action, self.train_q_net.num_actions_executed[action])
        print(str_info)

        vis.line(
            X=[total_steps], Y=[episode_reward], win=episode_reward_plt, name="Episode Reward", update="append",
            opts=dict(title='Episode Reward', showlegend=False)
        )
        vis.line(
            X=[total_steps], Y=[avg_episode_reward], win=avg_episode_reward_plt, name="Average Episode Reward", update="append",
            opts=dict(title='Average Episode Reward', showlegend=False)
        )
        vis.line(
            X=[total_steps], Y=[episode_loss], win=episode_loss_plt, name="Episode Loss", update="append",
            opts=dict(title='Episode Loss', showlegend=False)
        )


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
    dqn_agent.print_q_network()
    dqn_agent.learn()
    dqn_agent.save_model()

    dqn_agent2 = DqnAgent(env)
    dqn_agent2.load_model()
    execution(env, dqn_agent2)


if __name__ == "__main__":
    #
    main()
    # tensorboard --logdir 'logs/dqn/'
    # CARPOLE
    # python chapter_10/dqn.py --learning_rate=0.005 --epsilon_init=1.0 --epsilon_min=0.1 --replay_memory_capacity=8192 --target_net_update_freq=500 --epsilon_decay_end_step=15000 --max_steps=30000