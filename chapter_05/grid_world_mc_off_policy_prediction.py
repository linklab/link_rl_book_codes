# 사용 패키지 임포트
import numpy as np
from environments.gridworld import GridWorld
from utils.util import softmax

GRID_HEIGHT = 4
GRID_WIDTH = 4
NUM_ACTIONS = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
DISCOUNT_RATE = 1.0
MAX_EPISODES = 10000


class OffPolicyMonteCarloPrediction:
    def __init__(self, env):
        self.env = env

        self.max_iteration = MAX_EPISODES

        self.terminal_states = [(0, 0), (4, 4)]

        self.state_action_values = self.generate_initial_q_value()
        self.target_policy = self.generate_initial_random_policy()
        self.C = self.generate_initial_importance_sampling_weight_sum()

    # 비어있는 행동 가치 함수를 0으로 초기화하며 생성함
    def generate_initial_q_value(self):
        state_action_values = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                for action in range(NUM_ACTIONS):
                    state_action_values[((i, j), action)] = 0.0

        return state_action_values

    # 초기 중요도 샘플링 비율 가중치 생성
    def generate_initial_importance_sampling_weight_sum(self):
        C = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                for action in range(NUM_ACTIONS):
                    C[((i, j), action)] = 0.0
        return C

    # 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
    # 초기에 각 행동의 선택 확률은 모두 같음
    def generate_initial_random_policy(self):
        policy = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                actions = []
                prob = []
                for action in range(NUM_ACTIONS):
                    actions.append(action)
                    prob.append(0.25)
                policy[(i, j)] = (actions, prob)

        return policy

    def generate_any_policy(self):
        behavior_policy = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                r = sorted(np.random.sample(size=3)) # Return random floats in [0.0, 1.0)
                actions = [act for act in range(NUM_ACTIONS)]
                prob = [r[0], r[1] - r[0], r[2] - r[1], 1 - r[2]]
                behavior_policy[(i, j)] = (actions, prob)

        return behavior_policy

    # 환경에서 행위 정책을 통해 에피소드(현재 상태, 행동, 다음 상태, 보상)를 생성함
    def generate_episode(self, behavior_policy):
        episode = []
        visited_state_actions = []

        # i = random.randrange(GRID_HEIGHT)
        # j = random.randrange(GRID_WIDTH)
        # initial_state = (i, j)

        initial_state = (1, 1)

        self.env.moveto(initial_state)
        state = initial_state

        done = False
        while not done:
            actions, prob = behavior_policy[state]
            action = np.random.choice(actions, size=1, p=prob)[0]
            next_state, reward, done, _ = self.env.step(action)

            episode.append(((state, action), reward))
            visited_state_actions.append((state, action))

            state = next_state

        return episode, visited_state_actions

    # 첫 방문 행동 가치 MC 추정 함수
    def every_visit_mc_prediction(self, episode, behavior_policy):
        G = 0
        W = 1
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = DISCOUNT_RATE * G + reward
            self.C[(state, action)] += W
            self.state_action_values[(state, action)] += (G - self.state_action_values[(state, action)]) * W / self.C[(state, action)]
            if self.target_policy[state][1][action] == 0.0:
                print(state, self.target_policy[state][1], action, idx, self.C[(state, action)])
                break
            W = W * self.target_policy[state][1][action] / behavior_policy[state][1][action]


    # 탐험적 시작 전략 기반의 몬테카를로 방법 함수
    def off_policy_prediction(self):
        iter_num = 0

        print("[[[ Off-policy MC 예측 시작! ]]]")
        while iter_num < self.max_iteration:
            iter_num += 1

            print("*** 행위 정책 생성 ***")
            behavior_policy = self.generate_any_policy()

            print("*** 에피소드 생성 ***")
            episode, _ = self.generate_episode(behavior_policy)

            print("*** MC 예측 수행 ***")
            self.every_visit_mc_prediction(episode, behavior_policy)

            print("*** 총 반복 수: {0} ***".format(iter_num))

            print()

        print("[[[ MC 예 종료! ]]]\n\n")


def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_state=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    MC = OffPolicyMonteCarloPrediction(env)
    MC.off_policy_prediction()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print("({0}, {1}):".format(i, j))
            for action in range(NUM_ACTIONS):
                print("  Action {0}: {1:5.2f}".format(
                    env.action_space.ACTION_SYMBOLS[action],
                    MC.state_action_values[((i, j), action)]
                ))
        print()

    print()


if __name__ == "__main__":
    main()
