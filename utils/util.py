import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


def softmax(x):
    B = np.exp(x - np.max(x))
    C = np.sum(B)
    return B/C


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_grid_world_image(values, filename, grid_height, grid_width):
    state_values = np.zeros((grid_height, grid_width))
    for i in range(grid_height):
        for j in range(grid_width):
            state_values[(i, j)] = values[(i, j)]

    state_values = np.round(state_values, decimals=2)

    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = state_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for (i, j), val in np.ndenumerate(state_values):
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(state_values)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    ax.add_table(table)

    plt.savefig(filename)
    plt.close()


def print_grid_world_policy(env, policy):
    with np.printoptions(precision=2, suppress=True):
        for i in range(env.HEIGHT):
            for j in range(env.WIDTH):
                if (i, j) not in env.observation_space.TERMINAL_STATES:
                    print(
                        "({0}, {1}): UP, DOWN, LEFT, RIGHT".format(i, j),
                        policy[(i, j)][1],
                        env.action_space.ACTION_SYMBOLS[np.argmax(policy[(i, j)][1])]
                    )
            print()


def almost_equals(a, b, decimal=6):
    try:
        np.testing.assert_almost_equal(a, b, decimal=decimal)
    except AssertionError:
        return False
    return True