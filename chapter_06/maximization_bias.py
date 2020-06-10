import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import matplotlib as mpl

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 두 개의 정상 상태
STATE_A = 0
STATE_B = 1

# 두 개의 종료 상태
STATE_T1 = 2
STATE_T2 = 3

# 시작 상태 지정
STATE_START = STATE_A

# 상태 A에서의 두 행동
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1

EPSILON = 0.1

# 스텝 사이즈
ALPHA = 0.1

# 감가율
GAMMA = 1.0

# 상태 B에서의 가능한 행동 개수 - 10개
ACTIONS_B = range(0, 20)

# 각 상태에서의 가능한 행동
STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]

# 각 4개의 상태에서의 각 행동마다의 Q값을 지니는 Q-테이블 (초기 Q 값은 0)
INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1), np.zeros(1)]

# 상태 전이 관계 설정
TRANSITION = [
    [STATE_T2, STATE_B],            # STATE_A에서 RIGHT 수행하면 STATE_T2, LEFT 수행하면 STATE_B
    [STATE_T1] * len(ACTIONS_B)     # STATE_B에서 10개의 행동 중 하나를 수행하면 STATE_T1
]


# E-Greedy 정책 기반 행동 선택
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(STATE_ACTIONS[state])
    else:
        values_ = q_value[state]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# 상태, 행동 쌍 수행에 대한 보상
def get_reward(state, action):
    if state == STATE_A and action == ACTION_A_RIGHT:
        return 0
    elif state == STATE_A and action == ACTION_A_LEFT:
        return 0

    return np.random.normal(-0.1, 1)


# 만약 두 개의 Q 테이블이 주어지면 더블 Q-러닝이 수행되고, 한 개의 Q 테이블이 주어지면 Q-러닝이 수행된다.
def q_learning(q1, q2=None):
    state = STATE_START

    # 상태 A에서 'Left' 행동 수행 횟수
    left_count = 0

    while state != STATE_T1 and state != STATE_T2:
        if q2 is None:
            action = choose_action(state, q1)
        else:
            # Q1과 Q2를 합친 테이블을 기반으로 행동을 선택
            action = choose_action(state, [item1 + item2 for item1, item2 in zip(q1, q2)])

        if state == STATE_A and action == ACTION_A_LEFT:
            left_count += 1

        reward = get_reward(state, action)
        next_state = TRANSITION[state][action]

        if q2 is None:
            active_q = q1
            target = np.max(active_q[next_state])
        else:
            if np.random.binomial(1, 0.5) == 1:
                active_q = q1
                target_q = q2
            else:
                active_q = q2
                target_q = q1

            best_action = np.random.choice(
                [action_ for action_, value_ in enumerate(active_q[next_state]) if value_ == np.max(active_q[next_state])]
            )

            target = target_q[next_state][best_action]

        # Q-러닝 갱신 수행
        active_q[state][action] += ALPHA * (reward + GAMMA * target - active_q[state][action])
        state = next_state

    return left_count


def main():
    runs = 500
    episodes = 300
    left_counts_q = np.zeros((runs, episodes))
    left_counts_double_q = np.zeros((runs, episodes))

    for run in tqdm(range(runs)):
        q = copy.deepcopy(INITIAL_Q)
        q1 = copy.deepcopy(INITIAL_Q)
        q2 = copy.deepcopy(INITIAL_Q)

        for ep in range(episodes):
            left_counts_q[run, ep] = q_learning(q)
            left_counts_double_q[run, ep] = q_learning(q1, q2)

    left_counts_q = left_counts_q.mean(axis=0)
    left_counts_double_q = left_counts_double_q.mean(axis=0)

    plt.plot(left_counts_q, label='Q-러닝')
    plt.plot(left_counts_double_q, label='더블 Q-러닝')
    plt.plot(np.ones(episodes) * 0.05, label='최적', linestyle="dashed")
    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.xlabel('에피소드')
    plt.ylabel("A에서 'Left' 행동 선택 비율")
    plt.legend()

    plt.savefig('images/maximization_bias_{0}.png'.format(len(ACTIONS_B)))
    plt.close()


if __name__ == '__main__':
    main()