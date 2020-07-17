import numpy as np
import matplotlib.pyplot as plt

from chapter_08.dyna_q_common import DynaQParams, SimpleModel, dyna_q, draw_q_value_image
from environments.maze import Maze


def maze_dyna_q():
    dyna_maze = Maze() # 미로 환경 객체 구성
    dyna_params = DynaQParams()

    print(dyna_maze)

    episodes = 30
    planning_steps = [0, 3, 30]
    steps = np.zeros((len(planning_steps), episodes))

    for run in range(dyna_params.runs):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.max_planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)

            # Dyna-Q를 위한 단순 모델 생성
            model = SimpleModel()
            for episode in range(episodes):
                steps_, rewards_ = dyna_q(q_value, model, dyna_maze, dyna_params)
                steps[i, episode] += steps_
                # print('run: {0}, planning step: {1}, episode: {2}, steps: {3}, reward: {4}'.format(
                #     run, planning_step, episode, steps_, rewards_
                # ))
                if run == 0 and planning_step in [0, 30] and episode in [0, 1]:
                    draw_q_value_image(dyna_maze, q_value, run, planning_step, episode)

    # 총 수행 횟수에 대한 평균 값 산출
    steps /= dyna_params.runs

    linestyles = ['-', '--', ':']
    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], linestyle=linestyles[i], label='계획시 스텝 수행 횟: {0}'.format(planning_steps[i]))
    plt.xlabel('에피소드')
    plt.ylabel('에피소드당 진행 스텝수')
    plt.legend()
    plt.savefig('images/maze_dyna_q.png')
    plt.close()


if __name__ == '__main__':
    maze_dyna_q()
