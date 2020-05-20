# 사용 패키지 임포트
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.table import Table
from environments.gridworld import GridWorld

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(image):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for (i, j), val in np.ndenumerate(image):
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(image)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    ax.add_table(table)

# 환경에서 무작위로 에피소드(현재 상태, 행동, 다음 상태, 보상)를 생성함
def generate_random_episode(env):
    episode = []
    visited_states = []

    i = random.randrange(GRID_HEIGHT)
    j = random.randrange(GRID_WIDTH)
    initial_state = (i, j)
    env.moveto(initial_state)

    episode.append((initial_state, -1))
    visited_states.append(initial_state)

    done = False
    while not done:
        # 상태에 관계없이 항상 4가지 행동 중 하나를 선택하여 수행
        action = random.randrange(env.action_space.num_actions)

        state, reward, done, _ = env.step(action)

        if not done:
            episode.append((state, reward))
            visited_states.append(state)

    return episode, visited_states


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            state = (i, j)
            returns[state] = list()

    for i in range(num_iter):
        episode, visited_states = generate_random_episode(env)

        G = 0
        for idx, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            state_value_prediction_conditions = [
                state not in visited_states[:len(visited_states) - idx - 1],
                state not in TERMINAL_STATES
            ]

            if all(state_value_prediction_conditions):
                returns[state].append(G)
                state_values[state] = np.mean(returns[state])

    return state_values, returns


# 모든 방문 행동 가치 MC 예측
def every_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            state = (i, j)
            returns[state] = list()

    for i in range(num_iter):
        episode, _ = generate_random_episode(env)

        G = 0
        for idx, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            state_value_prediction_conditions = [
                state not in TERMINAL_STATES
            ]

            if all(state_value_prediction_conditions):
                returns[state].append(G)
                state_values[state] = np.mean(returns[state])

    return state_values, returns


if __name__ == "__main__":
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

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

    values, returns = first_visit_mc_prediction(env, 1.0, 10000)
    print(values)
    draw_image(np.round(values, decimals=2))
    plt.savefig('images/first_visit_mc_state_values.png')
    plt.close()

    print()
    values, returns = every_visit_mc_prediction(env, 1.0, 10000)
    print(values)
    draw_image(np.round(values, decimals=2))
    plt.savefig('images/every_visit_mc_state_values.png')
    plt.close()
