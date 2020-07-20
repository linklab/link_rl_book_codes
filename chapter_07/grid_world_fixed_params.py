import numpy as np
import os
import random
from environments.gridworld import GridWorld
from utils.util import draw_grid_world_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]

# 감가율
GAMMA = 1


# n-스텝 TD 방법
# @value: 본 함수에 의하여 갱신될 각 상태의 가치
# @n: n-스텝 TD 방법의 n
# @alpha: 스텝 사이즈
def temporal_difference(env, value, n, alpha):
    state = env.reset() # exploring start

    # 에피소드에 대하여 상태와 보상을 저장하는 배열
    states = [state]
    rewards = [0]

    # 타임 스텝
    time_step = 0

    # 이 에피소드의 길이
    T = float('inf')
    while True:
        if time_step < T:
            # 행동을 임의로 선택하여 다음 상태 결정
            action = random.randrange(env.action_space.num_actions)

            next_state, reward, done, _ = env.step(action)

            # 다음 타임 스텝에서의 보상과 상태 저장
            rewards.append(reward)
            states.append(next_state)

            if next_state in TERMINAL_STATES:
                T = time_step + 1

        # 갱신을 수행할 타임 스텝 결정
        tau = time_step - n + 1

        if tau >= 0:
            returns = 0.0

            # 대응되는 누적 보상(또는 이득)
            for i in range(tau + 1, min(tau + n, T) + 1):
                returns += pow(GAMMA, i - tau - 1) * rewards[i]

            # 누적 보상(또는 이득)에 상태 가치 추가
            if tau + n < T:
                returns += pow(GAMMA, n) * value[states[(tau + n)]]

            state_to_update = states[tau]

            # 상태 가치 갱신
            if state_to_update not in TERMINAL_STATES:
                value[state_to_update] += alpha * (returns - value[state_to_update])

        if tau == T - 1:
            break
        else:
            # 다음 타임 스텝
            time_step += 1
            state = next_state


def main():
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,    # exploring start
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    # n-스텝
    n = 3

    # 스텝 사이즈
    alpha = 0.2

    # 가 수행당 1번의 에피소드 수행
    episodes = 1000

    values = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            values[(i, j)] = 0.0

    for ep in range(episodes):
        temporal_difference(env, values, n, alpha)

    draw_grid_world_image(values, 'images/grid_world_fixed_params.png', GRID_HEIGHT, GRID_WIDTH)


if __name__ == '__main__':
    main()
