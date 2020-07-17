import random
import sys
import time

import gym
from PIL import Image
from gym.spaces import Box, Discrete
import cv2
import tensorflow as tf
import numpy as np

PONG_UP_ACTION = 2
PONG_DOWN_ACTION = 5

np.set_printoptions(threshold=sys.maxsize)


class PongWrappingEnv:
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.observation_space = Box(low=0, high=1, shape=(80, 80, 1))
        self.action_space = Discrete(n=2)

    def downsample(self, observation):
        # 오리지널 observation.shape --> (210, 160,3)

        # crop - 위에서 부터 35 픽셀 라인 제거, 아래에서 25 픽셀 라인 제거 (점수와 경계선 제거)
        observation = observation[35:185]

        # 그레이스케일로 변환
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # (80 * 80)으로 이미지 변환
        observation = cv2.resize(observation, (80, 80), interpolation=cv2.INTER_AREA)

        # img = Image.fromarray(observation, 'L')
        # img.show()

        observation = observation[:, :, None]

        observation = observation / 255

        return tf.cast(observation, dtype=tf.float32)

    def get_skipped_frames(self, action=None, reset=False, count=4):
        if reset:
            observation = self.env.reset()
            for _ in range(count - 1):
                action_ = self.env.action_space.sample()
                observation, _, _, _ = self.env.step(action=action_)
            observation = self.downsample(observation)
            return observation
        else:
            for _ in range(count - 1):
                action_ = self.env.action_space.sample()
                self.env.step(action=action_)
            observation, reward, done, info = self.env.step(action=action)
            observation = self.downsample(observation)
            return observation, reward, done, info

    def reset(self):
        # observation = self.get_skipped_frames(reset=True)
        observation = self.env.reset()
        observation = self.downsample(observation)
        return observation

    def step(self, action):
        # observation, reward, done, info = self.get_skipped_frames(action=action)
        observation, reward, done, info = self.env.step(action=action)
        observation = self.downsample(observation)
        return observation, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


if __name__ == "__main__":
    env = PongWrappingEnv()
    state = env.reset()
    done = False
    while not done:
        env.render()

        action = random.randint(0, env.action_space.n - 1)
        if action == 0:
            action = PONG_UP_ACTION
        elif action == 1:
            action = PONG_DOWN_ACTION
        else:
            raise ValueError()


        next_state, reward, done, info = env.step(action)
        print("State: {0}, Action: {1}, Next State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
            state.shape, action, next_state.shape, reward, done, info
        ))

        state = next_state
        time.sleep(0.25)
