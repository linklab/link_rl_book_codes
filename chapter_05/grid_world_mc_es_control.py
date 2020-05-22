# 사용 패키지 임포트
import numpy as np
import random
from environments.gridworld import GridWorld

GRID_HEIGHT = 4
GRID_WIDTH = 4
NUM_ACTIONS = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
DISCOUNT_RATE = 1.0
THETA_1 = 0.0001
THETA_2 = 0.0001
MAX_EPISODES = 5


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


class MonteCarloControl:
    def __init__(self, env):
        self.env = env

        self.max_iteration = MAX_EPISODES

        self.terminal_states = [(0, 0), (4, 4)]

        self.action_state_values = None
        self.policy = self.generate_random_policy()

    # 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
    # 초기에 각 행동의 선택 확률은 모두 같음
    def generate_random_policy(self):
        policy = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                actions = []
                prob = []
                for action in range(self.env.action_space.num_actions):
                    actions.append(action)
                    prob.append(0.25)
                policy[(i, j)] = (actions, prob)

        return policy

    # 환경에서 무작위로 에피소드(현재 상태, 행동, 다음 상태, 보상)를 생성함
    def generate_random_episode(self):
        episode = []
        visited_state_actions = []

        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        initial_state = (i, j)
        self.env.moveto(initial_state)

        state = initial_state
        done = False
        while not done:
            actions, prob = self.policy[state]
            action = np.random.choice(actions, size=1, p=prob)[0]
            next_state, reward, done, _ = self.env.step(action)

            episode.append(((state, action), reward))
            visited_state_actions.append((state, action))

            state = next_state

        return episode, visited_state_actions

    # 탐욕적인 정책을 생성함
    def generate_greedy_policy(self, state_action_values):
        new_policy = dict()

        is_policy_stable = True

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if (i, j) in TERMINAL_STATES:
                    actions = []
                    action_probs = []
                    for action in range(self.env.action_space.num_actions):
                        actions.append(action)
                        action_probs.append(0.25)
                    new_policy[(i, j)] = (actions, action_probs)
                else:
                    actions = []
                    q_values = []
                    for action in self.env.action_space.ACTIONS:
                        actions.append(action)
                        q_values.append(state_action_values[((i, j), action)])

                    new_policy[(i, j)] = (actions, softmax(q_values))

        error = 0.0
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                error += np.sum(np.absolute(np.array(self.policy[(i, j)][1]) - np.array(new_policy[(i, j)][1])))

        if error > THETA_2:
            is_policy_stable = False

        self.policy = new_policy

        return is_policy_stable, error


    # 첫 방문 행동 가치 MC 추정 함수
    def first_visit_mc_prediction(self, num_iter):
        # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
        state_action_values = dict()
        returns = dict()
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                for action in range(NUM_ACTIONS):
                    state_action_values[((i, j), action)] = 0.0
                    returns[((i, j), action)] = list()

        for i in range(num_iter):
            if i % 100 == 0:
                print(i, end=" ", flush=True)
            episode, visited_state_actions = self.generate_random_episode()

            G = 0
            for idx, ((state, action), reward) in enumerate(reversed(episode)):
                G = DISCOUNT_RATE * G + reward

                value_prediction_conditions = [
                    (state, action) not in visited_state_actions[:len(visited_state_actions) - idx - 1],
                    state not in TERMINAL_STATES
                ]

                if all(value_prediction_conditions):
                    returns[(state, action)].append(G)
                    state_action_values[(state, action)] = np.mean(returns[(state, action)])
        print()

        return state_action_values


    # 탐험적 시작 전략 기반의 몬테카를로 방법 함수
    def first_visit_mc_exploring_control_starts(self):
        iter_num = 0

        # 정책의 안정성 검증
        is_policy_stable = False

        print("[[[ MC 제어 반복 시작! ]]]")
        while not is_policy_stable and iter_num < self.max_iteration:
            print("*** MC 예측 수행 ***")
            new_q = self.first_visit_mc_prediction(num_iter=3000)
            is_policy_stable, error = self.generate_greedy_policy(state_action_values=new_q)
            print("*** 정책 개선 [에러 값: {0:7.5f}] ***".format(error))

            iter_num += 1
            print("*** 정책의 안정(Stable) 여부: {0}, 반복 횟수: {1} ***\n".format(
                is_policy_stable,
                iter_num
            ))

        print("[[[ MC 제어 반복 종료! ]]]\n\n")


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

    MC = MonteCarloControl(env)
    MC.first_visit_mc_exploring_control_starts()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print(
                i, j,
                ": UP, DOWN, LEFT, RIGHT",
                MC.policy[(i, j)][1]
            )
        print()


if __name__ == "__main__":
    main()
