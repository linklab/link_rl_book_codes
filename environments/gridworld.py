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
            height=5, width=5,
            start_state=(0, 0),
            terminal_states=[(4, 4)],
            transition_reward=0.0,
            terminal_reward=1.0,
            outward_reward=-1.0,
            warm_hole_states=None
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

        for state in terminal_states:     # 터미널 스테이트 제거
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
        self.observation_space.TERMINAL_STATES = terminal_states

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward
        self.outward_reward = outward_reward

        self.warm_hole_states = warm_hole_states
        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def is_warm_hole_state(self, state):
        i, j = state

        if self.warm_hole_states is not None and len(self.warm_hole_states) > 0:
            for warm_hole_info in self.warm_hole_states:
                warm_hole_state = warm_hole_info[0]
                if i == warm_hole_state[0] and j == warm_hole_state[1]:
                    return True
        return False

    def get_next_state_warm_hole(self, state):
        i, j = state
        next_state = None

        for warm_hole_info in self.warm_hole_states:
            warm_hole_state = warm_hole_info[0]
            warm_hole_prime_state = warm_hole_info[1]

            if i == warm_hole_state[0] and j == warm_hole_state[1]:
                next_state = warm_hole_prime_state
                break
        return next_state

    def get_reward_warm_hole(self, state):
        i, j = state
        reward = None

        for warm_hole_info in self.warm_hole_states:
            warm_hole_state = warm_hole_info[0]
            warm_hole_reward = warm_hole_info[2]

            if i == warm_hole_state[0] and j == warm_hole_state[1]:
                reward = warm_hole_reward
                break

        return reward

    def get_next_state(self, state, action):
        i, j = state

        if self.is_warm_hole_state(state):
            next_state = self.get_next_state_warm_hole(state)
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

        if self.is_warm_hole_state(state):
            reward = self.get_reward_warm_hole(state)
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
        next_i, next_j = self.get_next_state(state=self.current_state, action=action)

        reward = self.get_reward(self.current_state, (next_i, next_j))

        self.current_state = (next_i, next_j)

        if self.current_state in self.observation_space.TERMINAL_STATES:
            done = True
        else:
            done = False

        return (next_i, next_j), reward, done, None

    def render(self, mode='human'):
        print(self.__str__())

    def __str__(self):
        gridworld_str = ""
        for i in range(self.HEIGHT):
            gridworld_str += "-------------------------------\n"

            for j in range(self.WIDTH):
                if (i, j) == self.observation_space.START_STATE:
                    gridworld_str += "|  {0}  ".format("S")
                elif (i, j) in self.observation_space.TERMINAL_STATES:
                    gridworld_str += "|  {0}  ".format("G")
                elif self.current_state[0] == i and self.current_state[1] == j:
                    gridworld_str += "|  {0}  ".format("*")
                else:
                    gridworld_str += "|     "
            gridworld_str += "|\n"

            for j in range(self.WIDTH):
                gridworld_str += "|({0},{1})".format(i, j)

            gridworld_str += "|\n"

        gridworld_str += "-------------------------------\n"
        return gridworld_str

if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)
