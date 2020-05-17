import time

import gym

# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------


class GridWorld(gym.Env):
    def __init__(
            self,
            height=5,
            width=5,
            start_state=(0, 0),
            terminal_state=[(4, 4)],
            transition_reward=0.0,
            terminal_reward=1.0,
            outward_reward=-1.0,
            warmhole_states=None
    ):
        self.__version__ = "0.0.1"

        self.observation_space = gym.spaces.MultiDiscrete([height, width])
        self.action_space = gym.spaces.Discrete(4)

        # 그리드월드의 세로 길이
        self.HEIGHT = height

        # 그리드월드의 가로 길이
        self.WIDTH = width

        self.observation_space.STATES = []
        self.observation_space.num_states = self.WIDTH * self.HEIGHT

        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.observation_space.STATES.append((i, j))

        for state in terminal_state:     # 터미널 스테이트 제거
            self.observation_space.STATES.remove(state)

        # 모든 가능한 행동
        self.action_space.ACTION_UP = 0
        self.action_space.ACTION_DOWN = 1
        self.action_space.ACTION_LEFT = 2
        self.action_space.ACTION_RIGHT = 3
        self.action_space.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"]
        self.action_space.ACTIONS = [
            self.action_space.ACTION_UP,
            self.action_space.ACTION_DOWN,
            self.action_space.ACTION_LEFT,
            self.action_space.ACTION_RIGHT
        ]
        self.action_space.num_actions = len(self.action_space.ACTIONS)

        # 시작 상태 위치
        self.observation_space.START_STATE = start_state

        # 종료 상태 위치
        self.observation_space.TERMINAL_STATES = terminal_state

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward
        self.outward_reward = outward_reward

        self.warmhole_states = warmhole_states
        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def is_warmhole_state(self, state):
        i, j = state

        if self.warmhole_states is not None and len(self.warmhole_states) > 0:
            for warmhole_info in self.warmhole_states:
                warmhole_state = warmhole_info[0]
                if i == warmhole_state[0] and j == warmhole_state[1]:
                    return True
        return False

    def get_next_state_warmhole(self, state):
        i, j = state
        next_state = None

        for warmhole_info in self.warmhole_states:
            warmhole_state = warmhole_info[0]
            warmhole_prime_state = warmhole_info[1]

            if i == warmhole_state[0] and j == warmhole_state[1]:
                next_state = warmhole_prime_state
                break
        return next_state

    def get_reward_warmhole(self, state):
        i, j = state
        reward = None

        for warmhole_info in self.warmhole_states:
            warmhole_state = warmhole_info[0]
            warmhole_reward = warmhole_info[2]

            if i == warmhole_state[0] and j == warmhole_state[1]:
                reward = warmhole_reward
                break

        return reward

    def get_next_state(self, state, action):
        i, j = state

        if self.is_warmhole_state(state):
            next_state = self.get_next_state_warmhole(state)
            next_i = next_state[0]
            next_j = next_state[1]
        elif (i, j) in self.observation_space.TERMINAL_STATES:
            next_i = i
            next_j = j
        else:
            if action == self.action_space.ACTION_UP:
                next_i = max(i - 1, 0)
                next_j = j
            elif action == self.action_space.ACTION_DOWN:
                next_i = min(i + 1, self.HEIGHT - 1)
                next_j = j
            elif action == self.action_space.ACTION_LEFT:
                next_i = i
                next_j = max(j - 1, 0)
            elif action == self.action_space.ACTION_RIGHT:
                next_i = i
                next_j = min(j + 1, self.WIDTH - 1)
            else:
                raise ValueError()

        return next_i, next_j

    def get_reward(self, state, next_state):
        i, j = state
        next_i, next_j = next_state

        if self.is_warmhole_state(state):
            reward = self.get_reward_warmhole(state)
        else:
            if (next_i, next_j) in self.observation_space.TERMINAL_STATES:
                reward = self.terminal_reward
            else:
                if i == next_i and j == next_j:
                    reward = self.outward_reward
                else:
                    reward = self.transition_reward

        return reward

    def get_state_action_probability(self, state, action):
        next_i, next_j = self.get_next_state(state, action)

        reward = self.get_reward(state, (next_i, next_j))
        prob = 1.0

        return (next_i, next_j), reward, prob

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        i, j = self.current_state

        next_i, next_j = self.get_next_state(state=self.current_state, action=action)

        reward = self.get_reward(self.current_state, (next_i, next_j))

        self.current_state = (next_i, next_j)

        if self.current_state in self.observation_space.TERMINAL_STATES:
            done = True
        else:
            done = False

        return (next_i, next_j), reward, done, None

    def render(self, mode='human'):
        for i in range(self.HEIGHT):
            print("-------------------------------")

            for j in range(self.WIDTH):
                if self.current_state[0] == i and self.current_state[1] == j:
                    print("|  {0}  ".format("*"), end="")
                else:
                    print("|     ".format("*"), end="")
            print("|")

            for j in range(self.WIDTH):
                print("|({0},{1})".format(i, j), end="")
            print("|")

        print("-------------------------------\n")


if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)
