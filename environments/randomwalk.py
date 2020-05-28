import time

import gym

# -------------------------------
# T1 0 1 2 3 4 T2
# -------------------------------


class RandomWalk(gym.Env):
    def __init__(
            self,
            num_internal_states=5,
            transition_reward=0.0,
            left_terminal_reward=0.0,
            right_terminal_reward=1.0
    ):
        self.__version__ = "0.0.1"

        self.num_internal_states = num_internal_states
        self.observation_space = gym.spaces.Discrete(num_internal_states + 2)
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space.num_states = num_internal_states + 2
        self.observation_space.STATES = [i for i in range(num_internal_states)]
        self.observation_space.TERMINAL_STATES = ['T1', 'T2']

        # 모든 가능한 행동
        self.action_space.ACTION_LEFT = 0
        self.action_space.ACTION_RIGHT = 1
        self.action_space.ACTION_SYMBOLS = ["\u2190", "\u2192"]
        self.action_space.ACTIONS = [
            self.action_space.ACTION_LEFT,
            self.action_space.ACTION_RIGHT
        ]
        self.action_space.num_actions = len(self.action_space.ACTIONS)

        # 시작 상태 위치
        self.observation_space.START_STATE = self.observation_space.STATES[int(num_internal_states / 2)]

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.left_terminal_reward = left_terminal_reward

        self.right_terminal_reward = right_terminal_reward

        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def get_next_state(self, state, action):
        if state in self.observation_space.TERMINAL_STATES:
            next_state = state
        else:
            if action == self.action_space.ACTION_LEFT:
                if state == 0:
                    next_state = 'T1'
                else:
                    next_state = state - 1
            elif action == self.action_space.ACTION_RIGHT:
                if state == self.num_internal_states - 1:
                    next_state = 'T2'
                else:
                    next_state = state + 1
            else:
                raise ValueError()

        return next_state

    def get_reward(self, state, next_state):
        if next_state == 'T1':
            reward = self.left_terminal_reward
        elif next_state == 'T2':
            reward = self.right_terminal_reward
        else:
            reward = self.transition_reward

        return reward

    def get_state_action_probability(self, state, action):
        next_state = self.get_next_state(state, action)

        reward = self.get_reward(state, next_state)
        prob = 1.0

        return next_state, reward, prob

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        next_state = self.get_next_state(state=self.current_state, action=action)

        reward = self.get_reward(self.current_state, next_state)

        self.current_state = next_state

        if self.current_state in self.observation_space.TERMINAL_STATES:
            done = True
        else:
            done = False

        return next_state, reward, done, None

    def render(self, mode='human'):
        print(" T1 " + " ".join(["{0}".format(i) for i in range(self.num_internal_states)]) + " T2")
        if self.current_state in self.observation_space.STATES:
            blank = "    " + "  " * self.current_state
        elif self.current_state == 'T1':
            blank = " "
        elif self.current_state == 'T2':
            blank = "  " + "  " * (self.num_internal_states + 1)
        else:
            raise ValueError()

        print(blank + str(self.current_state), flush=True)
        print()


def main():
    env = RandomWalk()
    env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)


if __name__ == "__main__":
    main()
