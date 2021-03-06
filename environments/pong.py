# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
import random
import sys
import time
from collections import deque

import gym
from gym.spaces import Box, Discrete
import cv2
import numpy as np
import matplotlib.pyplot as plt

NUM_STACKED_FRAMES = 4
PONG_NOOP = 0
PONG_UP_ACTION = 2
PONG_DOWN_ACTION = 5

np.set_printoptions(threshold=sys.maxsize)


class PongWrappingEnv:
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.observation_space = Box(low=0, high=1, shape=(80, 80, 4))
        self.action_space = Discrete(n=2)
        self.env_name = 'pong'
        self.stacked_observations = deque(maxlen=NUM_STACKED_FRAMES)

    def downsample(self, observation):
        """
        오리지널 observation.shape: (210, 160, 3) --> 변환되는 observation.shape: (80, 80, 1)
        :param observation:
        :return:
        """

        # crop - 위에서 부터 35 픽셀 라인 제거, 아래에서 25 픽셀 라인 제거 (점수와 경계선 제거)
        observation = observation[35:185]

        # 그레이스케일로 변환
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # (80 * 80)으로 이미지 변환
        observation = cv2.resize(observation, (80, 80), interpolation=cv2.INTER_AREA)

        observation = np.array(observation).astype(np.float32) / 255.0

        # plt.imshow(observation, 'gray')
        # plt.show()

        return observation

    def reset(self):
        observation = self.env.reset()
        observation = self.downsample(observation)
        self.stacked_observations.append(observation)

        for _ in range(NUM_STACKED_FRAMES - 1):
            observation, reward, done, info = self.env.step(PONG_NOOP)
            observation = self.downsample(observation)
            self.stacked_observations.append(observation)

        stacked_observations = np.stack(self.stacked_observations, axis=2)

        return stacked_observations

    def step(self, action):
        if action == 0:
            action = PONG_UP_ACTION
        elif action == 1:
            action = PONG_DOWN_ACTION

        observation, reward, done, info = self.env.step(action=action)

        observation = self.downsample(observation)
        self.stacked_observations.append(observation)
        stacked_observations = np.stack(self.stacked_observations, axis=2)

        return stacked_observations, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


def main():
    env = PongWrappingEnv()
    state = env.reset()
    done = False
    while not done:
        env.render()

        action = random.randint(0, env.action_space.n - 1)

        next_state, reward, done, info = env.step(action)
        print("State: {0}, Action: {1}, Next State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
            state.shape, action, next_state.shape, reward, done, info
        ))

        state = next_state
        time.sleep(0.25)


if __name__ == "__main__":
    main()
