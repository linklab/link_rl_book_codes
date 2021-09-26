# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


class ReplayMemory:
    def __init__(self, buffer_limit=50000):
        self.memory = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.memory.append(transition)

    def size(self):
        return len(self.memory)

    def sample(self, n):
        mini_batch = random.sample(self.memory, n)
        observation_lst, action_lst, reward_lst, next_observation_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            observation, action, reward, next_observation, done_mask = transition
            observation_lst.append(observation)
            action_lst.append([action])
            reward_lst.append([reward])
            next_observation_lst.append(next_observation)
            done_mask_lst.append([done_mask])

        return torch.tensor(observation_lst, dtype=torch.float), torch.tensor(action_lst), \
               torch.tensor(reward_lst), torch.tensor(next_observation_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)


def q_learning(
        env, num_episodes=1000, learning_rate=0.0001, gamma=0.99,
        epsilon_start=0.2, epsilon_end=0.01, batch_size=32,
        train_step_interval=4, target_update_step_interval=100,
        print_episode_interval=10
):
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayMemory()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    episode_reward_list = []

    training_steps = 0
    last_episode_reward = 0

    total_step_idx = 0

    for i in range(num_episodes):
        # Linear decaying from epsilon_start to epsilon_end
        epsilon = max(epsilon_end, epsilon_start - (i / num_episodes))

        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        #env.render()
        # The Q-Table 알고리즘
        while True:
            total_step_idx += 1
            # 가장 Q값이 높은 action을 결정함

            action = q.get_action(torch.from_numpy(observation).float(), epsilon)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)
            #env.render()
            done_mask = 0.0 if done else 1.0

            memory.put(
                (observation, action, reward / 100.0, next_observation, done_mask)
            )

            if memory.size() > 2000 and total_step_idx % train_step_interval == 0:
                observation_t, action_t, reward_t, next_observation_t, done_mask_t = memory.sample(batch_size)

                q_out = q(observation_t)                                   # q_out.shape: (32, 2)
                q_a = q_out.gather(dim=1, index=action_t)                  # q_a.shape: (32, 2)

                q_prime_out = q_target(next_observation_t)                 # q_prime_out.shape: (32, 2)
                max_q_prime = q_prime_out.max(dim=1)[0].unsqueeze(dim=-1)  # max_q_prime.shape: (32, 1)

                target = reward_t + gamma * max_q_prime * done_mask_t
                loss = F.mse_loss(target, q_a)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_steps += 1  # Q 업데이트 횟수
                episode_reward_list.append(last_episode_reward)

            if total_step_idx % target_update_step_interval == 0:
                q_target.load_state_dict(q.state_dict())

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            if done:
                last_episode_reward = episode_reward
                break

        if i % print_episode_interval == 0 and i != 0:
            print("EPISODE: {0:3d}, EPISODE_REWARD: {1:5.1f}, SIZE_OF_REPLAY_BUFFER: {2:5d}, EPSILON: {3:.3f}".format(
                i, episode_reward, memory.size(), epsilon
            ))

    return training_steps, episode_reward_list


def q_testing(num_episodes, q):
    episode_reward_list = []

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        pass # 숙제

        episode_reward_list.append(episode_reward)

    return np.average(episode_reward_list), np.std(episode_reward_list)


def main_env_info():
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    env = gym.make('CartPole-v1')

    #####################
    # observation space #
    #####################
    # Observation:
    # Type: Box(4)
    # Num     Observation               Min                     Max
    # 0       Cart Position             -4.8                    4.8
    # 1       Cart Velocity             -Inf                    Inf
    # 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    # 3       Pole Angular Velocity     -Inf                    Inf
    print("*" * 80)
    print(env.observation_space)
    # print(env.observation_space.n)

    for i in range(10):
        print(env.observation_space.sample())
    print()

    ################
    # action space #
    ################
    # Actions:
    # Type: Discrete(2)
    # Num   Action
    # 0     Push cart to the left
    # 1     Push cart to the right
    print("*" * 80)
    print(env.action_space)
    print(env.action_space.n)
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # Starting State:
    # All observations are assigned a uniform random value in [-0.05..0.05]
    observation = env.reset()
    print(observation)

    # Reward:
    # Reward is 1 for every step taken, including the termination step
    action = 0  # LEFT
    next_observation, reward, done, info = env.step(action)

    # Observation = 1: move to grid number 1 (unchanged)
    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, Reward: {2}, Next Observation: {3}, Done: {4}, Info: {5}".format(
        observation, action, reward, next_observation, done, info
    ))

    observation = next_observation

    action = 1  # RIGHT
    next_observation, reward, done, info = env.step(action)

    # Observation = 5: move to grid number 5 (unchanged)
    print("Observation: {0}, Action: {1}, Reward: {2}, Next Observation: {3}, Done: {4}, Info: {5}".format(
        observation, action, reward, next_observation, done, info
    ))

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()

    actions = [0, 1] * 5
    for action in actions:
        next_observation, reward, done, info = env.step(action)
        print("Observation: {0}, Action: {1}, Reward: {2}, Next Observation: {3}, Done: {4}, Info: {5}".format(
            observation, action, reward, next_observation, done, info
        ))
        observation = next_observation

    env.close()


def main_q_learning():
    NUM_EPISODES = 1000
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 0.2
    EPSILON_END = 0.01
    BATCH_SIZE = 32
    TRAIN_STEP_INTERVAL = 4
    TARGET_UPDATE_STEP_INTERVAL = 100
    PRINT_EPISODE_INTERVAL = 10

    env = gym.make('CartPole-v1')

    training_steps, episode_reward_list = q_learning(
        env,
        NUM_EPISODES, LEARNING_RATE, GAMMA,
        EPSILON_START, EPSILON_END, BATCH_SIZE, TRAIN_STEP_INTERVAL,
        TARGET_UPDATE_STEP_INTERVAL, PRINT_EPISODE_INTERVAL
    )

    plt.plot(range(training_steps), episode_reward_list, color="Blue")
    plt.xlabel("training steps")
    plt.ylabel("episode reward")
    plt.show()


if __name__ == "__main__":
    main_env_info()
    main_q_learning()
