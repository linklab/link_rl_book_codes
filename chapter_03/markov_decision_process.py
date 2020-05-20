import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table
from environments.gridworld import GridWorld

# 이미지 저장 경로 확인 및 생성
if not os.path.exists('images/'):
    os.makedirs('images/')

GRID_HEIGHT = 5
GRID_WIDTH = 5

DISCOUNT_RATE = 0.9      # 감쇄율

A_POSITION = (0, 1)         # 임의로 지정한 특별한 상태 A 좌표
B_POSITION = (0, 3)         # 임의로 지정한 특별한 상태 B 좌표

A_PRIME_POSITION = (4, 1)   # 상태 A에서 행동시 도착할 위치 좌표
B_PRIME_POSITION = (2, 3)   # 상태 B에서 행동시 도착할 위치 좌표


# 기본 GridWorld 환경에 추가할 조건을 정의하는 함수
def unique_step_wormhole(state, action):
    if state == A_POSITION:
        return A_PRIME_POSITION, 10, False, {'exec': True}
    elif state == B_POSITION:
        return B_PRIME_POSITION, 5, False, {'exec': True}
    return state, None, False, {'exec': False}


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(state_values, filename):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = state_values.shape
    width, height = 1.0 / num_cols, 1.0 / num_rows

    # 렌더링할 이미지에 표 셀 추가
    for (i, j), val in np.ndenumerate(state_values):
        if val == 0.0:
            val = ''
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(state_values)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    axis.add_table(table)

    plt.savefig(filename, dpi=300)



# GRID_WORLD 첫번째 예제
def grid_world_state_values(env):
    ACTION_PROBABILITY = 0.25
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # value_function과 동일한 형태를 가지면서 값은 모두 0인 배열을 new_value_function에 저장
        new_value_function = np.zeros_like(value_function)

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                values = []
                # 주어진 상태에서 가능한 모든 행동들의 결과로 다음 상태들을 갱신
                for action in env.action_space.ACTIONS:
                    (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)

                    # Bellman-Equation, 벨만 방정식 적용
                    values.append(
                        ACTION_PROBABILITY * prob * (reward + DISCOUNT_RATE * value_function[next_i, next_j])
                    )

                new_value_function[i, j] = np.sum(values)

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(value_function - new_value_function)) < 1e-4:
            break

        # 가치 함수 갱신
        value_function = new_value_function

    return new_value_function


# GRID_WORLD 두번째 예제
def grid_world_optimal_values(env):
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # value_function과 동일한 형태를 가지면서 값은 모두 0인 배열을 new_value_function에 저장
        new_value_function = np.zeros_like(value_function)

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                values = []
                # 주어진 상태에서 가능한 모든 행동들의 결과로 다음 상태들을 갱신
                for action in env.action_space.ACTIONS:
                    (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)

                    # Bellman Optimality Equation, 벨만 최적 방정식 적용
                    values.append(
                        prob * (reward + DISCOUNT_RATE * value_function[next_i, next_j])
                    )

                # 새롭게 계산된 상태 가치 중 최대 상태 가치로 현재 상태의 가치 갱신
                new_value_function[i, j] = np.max(values)

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(new_value_function - value_function)) < 1e-4:
            break

        value_function = new_value_function

    return new_value_function


# MAIN
if __name__ == '__main__':
    if not os.path.exists('images/'):
        os.makedirs('images/')

    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    draw_image(np.round(value_function, decimals=0), 'images/empty_grid_world.png')

    # 5x5 맵 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_state=[],
        transition_reward=0,
        outward_reward=-1.0,
        warmhole_states=[(A_POSITION, A_PRIME_POSITION, 10.0), (B_POSITION, B_PRIME_POSITION, 5.0)]
    )

    env.reset()
    state_values = grid_world_state_values(env)
    print(state_values)
    draw_image(np.round(state_values, decimals=2), 'images/grid_world_state_values.png')

    print()

    env.reset()
    optimal_state_values = grid_world_optimal_values(env)
    print(optimal_state_values)
    draw_image(np.round(optimal_state_values, decimals=2), 'images/grid_world_optimal_values.png')
