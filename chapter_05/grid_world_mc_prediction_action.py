# 사용 패키지 임포트
import numpy as np
import random
from environments.gridworld import GridWorld

GRID_HEIGHT = 4
GRID_WIDTH = 4
NUM_ACTIONS = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]


def get_explorering_start_state():
    while True:
        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        if (i, j) not in TERMINAL_STATES:
            break
    return (i, j)


# 환경에서 무작위로 에피소드(현재 상태, 행동, 다음 상태, 보상)를 생성함
def generate_random_episode(env):
    episode = []
    visited_state_actions = []

    initial_state = get_explorering_start_state()
    env.moveto(initial_state)

    state = initial_state
    done = False
    while not done:
        # 상태에 관계없이 항상 4가지 행동 중 하나를 선택하여 수행
        action = random.randrange(env.action_space.num_actions)

        next_state, reward, done, _ = env.step(action)

        episode.append(((state, action), reward))
        visited_state_actions.append((state, action))

        state = next_state

    return episode, visited_state_actions


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values = dict()
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in range(NUM_ACTIONS):
                state_action_values[((i, j), action)] = 0.0
                returns[((i, j), action)] = list()

    for i in range(num_iter):
        episode, visited_state_actions = generate_random_episode(env)

        G = 0
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            value_prediction_conditions = [
                (state, action) not in visited_state_actions[:len(visited_state_actions) - idx - 1],
                state not in TERMINAL_STATES
            ]

            if all(value_prediction_conditions):
                returns[(state, action)].append(G)
                state_action_values[(state, action)] = np.mean(returns[(state, action)])

    return state_action_values, returns


# 모든 방문 행동 가치 MC 예측
def every_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values = dict()
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in range(NUM_ACTIONS):
                state_action_values[((i, j), action)] = 0.0
                returns[((i, j), action)] = list()

    for i in range(num_iter):
        episode, _ = generate_random_episode(env)

        G = 0
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            value_prediction_conditions = [
                state not in TERMINAL_STATES
            ]

            if all(value_prediction_conditions):
                returns[(state, action)].append(G)
                state_action_values[(state, action)] = np.mean(returns[(state, action)])

    return state_action_values, returns

def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    state_action_values, returns = first_visit_mc_prediction(env, 1.0, 10000)
    print("First Visit")
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print("({0}, {1}):".format(i, j))
            for action in range(NUM_ACTIONS):
                print("  Action {0}: {1:5.2f}".format(
                    env.action_space.ACTION_SYMBOLS[action],
                    state_action_values[((i, j), action)]
                ))
        print()

    print()

    state_action_values, returns = every_visit_mc_prediction(env, 1.0, 10000)
    print("Every Visit")
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print("({0}, {1}):".format(i, j))
            for action in range(NUM_ACTIONS):
                print("  Action {0}: {1:5.2f}".format(
                    env.action_space.ACTION_SYMBOLS[action],
                    state_action_values[((i, j), action)]
                ))
        print()

    print()


if __name__ == "__main__":
    main()
