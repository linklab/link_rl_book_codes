import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm
import heapq
from copy import deepcopy
import random

from chapter_08.dyna_q_common import DynaQParams, PriorityModel, SimpleModel, prioritized_sweeping, dyna_q
from environments.maze import Maze


# 일련의 상태-행동 쌍이 시작 상태에서 목적 상태까지 최적의 경로를 산출한는지 체크함
def check_path(q_values, dyna_maze):
    # 우선 목적 상태까지의 길이를 산출한다.
    # 주어진 기본 장애물에 대해서 최적 경로의 스텝 수는 15이다.
    # 1.4 는 좀 더 완화된 최적 경로 길이 허용 수준을 의미한다.
    max_steps = 15 * 1.4
    state = dyna_maze.START_STATE
    steps = 0
    while state not in dyna_maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        _, state = dyna_maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False

    return True


# mazes with different resolution
def prioritized_sweeping_maze_dyna_q():
    # get the original 6 * 10 dyna_maze
    dyna_maze = Maze()

    print(dyna_maze)

    # set up the parameters for each algorithm
    dyna_params = DynaQParams()
    dyna_params.max_planning_steps = 5
    dyna_params.alpha = 0.5
    dyna_params.gamma = 0.95

    params_prioritized = DynaQParams()
    params_prioritized.theta = 0.001
    params_prioritized.max_planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, dyna_params]

    # set up models for planning
    models = [PriorityModel, SimpleModel]
    method_names = ['Priority Sweeping', 'Dyna-Q']

    # build all the mazes
    methods = [prioritized_sweeping, dyna_q]

    runs = 20

    # track the # of backups
    backups = np.zeros((runs, len(models)))

    for run in range(0, runs):
        for i in range(0, len(method_names)):
            print('run %d, %s, dyna_maze size %d' % (
                run, method_names[i], dyna_maze.MAZE_HEIGHT * dyna_maze.MAZE_WIDTH)
            )

            # initialize the state action values
            q_value = np.zeros(dyna_maze.q_size)

            # track steps / backups for each episode
            steps = []

            # generate the model
            model = models[i]()

            # play for an episode
            while True:
                steps.append(methods[i](q_value, model, dyna_maze, params[i]))

                # check whether the (relaxed) optimal path is found
                if check_path(q_value, dyna_maze):
                    break

            # update the total steps / backups for this dyna_maze
            backups[run, i] = np.sum(steps)

    backups = backups.mean(axis=0)

    # # Dyna-Q performs several backups per step
    # backups[:] *= dyna_params.max_planning_steps + 1

    print(backups)

    for i in range(0, len(method_names)):
        plt.bar(range(len(backups)), backups, width=0.5)

    plt.xlabel('방법')
    plt.xticks(range(len(backups)), method_names)
    plt.ylabel('최적 경로 발견까지 수행한 역갱신 횟수')

    plt.savefig('images/prioritized_sweeping.png')
    plt.close()


if __name__ == '__main__':
    prioritized_sweeping_maze_dyna_q()
