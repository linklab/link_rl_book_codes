import numpy as np
import os
from environments.gridworld import GridWorld
from utils.util import draw_grid_world_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT - 1, GRID_WIDTH - 1)]
DISCOUNT_RATE = 1.0
THETA = 0.0001


# 상태 가치 함수를 계산하는 함수
def compute_state_value(env):
    ACTION_PROBABILITY = 0.25

    state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # 가치 함수의 값들이 수렴할 때까지 반복
    iter_num = 0
    while True:
        old_state_values = state_values.copy()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if (i, j) in TERMINAL_STATES:
                    state_values[i][j] = 0.0
                else:
                    values = []
                    for action in env.action_space.ACTIONS:
                        (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)

                        # Bellman-Equation, 벨만 방정식 적용
                        values.append(
                            ACTION_PROBABILITY * prob * (reward + DISCOUNT_RATE * old_state_values[next_i, next_j])
                        )

                    state_values[i][j] = np.sum(values)

        iter_num += 1

        # 갱신되는 값이 THETA_1(=0.0001)을 기준으로 수렴하는지 판정
        max_delta_value = abs(old_state_values - state_values).max()
        if max_delta_value < THETA:
            break

    return state_values, iter_num


def grid_world_policy_evaluation():
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

    # 수렴 시킨 상태 가치를 이미지로 저장하고 반복 횟수 반환 받음
    state_values, iteration = compute_state_value(env)

    print('정책 평가 --> 상태 가치 수렴: {} 회 반복'.format(iteration))
    print(state_values)

    draw_grid_world_image(
        np.round(state_values, decimals=2), 'images/state_values.png', GRID_HEIGHT, GRID_WIDTH
    )


# MAIN
if __name__ == '__main__':
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    grid_world_policy_evaluation()
