import numpy as np
import matplotlib.pyplot as plt

from chapter_08.dyna_q_common import DynaQParams, dyna_q, dyna_q_2
from environments.maze import Maze


# 장애물 위치가 변하는 미로 환경에서의 Dyna-Q 및 Dyna-Q+ 성능 비교
def changing_maze_dyna_q():
    # 장애물 구성 변경 미로 환경 구성
    changing_maze = Maze()
    original_obstacles = [(3, i) for i in range(0, 9)]
    changing_maze.obstacles = original_obstacles

    print(changing_maze)

    # 최적 경로를 가로 막는 새로운 장애물 구성
    new_obstacles = [(3, i) for i in range(1, 10)]

    # 최대 스텝 설정
    changing_maze.max_steps = 3000

    # 장애물 구성이 변경되는 스텝 설정
    obstacle_switch_time = 1500

    # 파라미터 설정
    dyna_params = DynaQParams()
    dyna_params.alpha = 1.0
    dyna_params.max_planning_steps = 10
    dyna_params.runs = 10

    # 시간 기반 모델에 사용할 파라미터
    dyna_params.time_weight = 0.0001

    # 훈련 후 누적 보상 획득
    cumulative_rewards = dyna_q_2(
        changing_maze, dyna_params, original_obstacles, new_obstacles, obstacle_switch_time
    )

    for i in range(len(dyna_params.methods)):
        plt.plot(cumulative_rewards[i, :], label=dyna_params.methods[i])

    plt.xlabel('누적 타임 스텝')
    plt.ylabel('누적 보상')
    plt.legend()

    plt.savefig('images/changing_maze_dyna_q.png')
    plt.close()


if __name__ == '__main__':
    changing_maze_dyna_q()
