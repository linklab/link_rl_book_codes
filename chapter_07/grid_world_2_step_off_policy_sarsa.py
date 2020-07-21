import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from environments.gridworld import GridWorld
from utils.util import print_grid_world_policy

plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]

# 초기 하이퍼파라미터 설정: ε=0.2, α=0.5, γ=0.98, n-스텝 = 3, 에피소드 수행 횟수 = 100
EPSILON = 0.2
ALPHA = 0.1
GAMMA = 0.98
MAX_EPISODES = 10000


# 행동-가치 함수 생성
def state_action_value(env):
    q = dict()
    for state in env.observation_space.STATES:
        for action in env.action_space.ACTIONS:
            q[(state, action)] = np.random.normal()
    return q


# ε-탐욕적 정책의 확률 계산 함수
def e_greedy(env, e, q, state):
    action_values = []
    prob = []
    for action in env.action_space.ACTIONS:
        action_values.append(q[(state, action)])

    for i in range(len(action_values)):
        if i == np.argmax(action_values):
            prob.append(1 - e + e/len(action_values))
        else:
            prob.append(e/len(action_values))
    return env.action_space.ACTIONS, prob


# ε-탐욕적 정책 생성 함수
def generate_e_greedy_policy(env, e, Q):
    policy = dict()
    for state in env.observation_space.STATES:
        policy[state] = e_greedy(env, e, Q, state)
    return policy


# ε-탐욕적 정책 생성 함수
def generate_greedy_policy(env, Q):
    policy = dict()
    for state in env.observation_space.STATES:
        policy[state] = e_greedy(env, 0.0, Q, state)
    return policy


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 각 행동의 선택 확률은 모두 같음
def generate_random_policy(env):
    policy = dict()
    for state in env.observation_space.STATES:
        actions = []
        prob = []
        for action in env.action_space.ACTIONS:
            actions.append(action)
            prob.append(0.25)
        policy[state] = (actions, prob)
    return policy


# n-스텝 SARSA 함수
def n_step_off_policy_td(env, Q, policy, b, n):
    episode_reward_list = []

    for episode in range(MAX_EPISODES):
        state = env.reset()     # exploring start
        action = np.random.choice(policy[state][0], p=policy[state][1])
        state_trace, action_trace, reward_trace = [state], [action], []

        # 타임 스텝
        time_step = 0

        # 이 에피소드의 길이
        T = float('inf')

        while True:
            if time_step < T:
                next_state, reward, done, _ = env.step(action)
                reward_trace.append(reward)

                if next_state in TERMINAL_STATES:
                    T = time_step + 1
                else:
                    state_trace.append(next_state)
                    next_action = np.random.choice(b[next_state][0], p=b[next_state][1])
                    action_trace.append(next_action)
                    action = next_action

            # 갱신을 수행할 타임 스텝 결정
            tau = time_step - n + 1

            if tau >= 0:     # update_state 시작위치부터 n개를 reward_trace[]에서 가져와야 하기 때문
                rho = 1
                for i in range(tau + 1, min([tau + n, T - 1]) + 1):
                    y = policy[state_trace[i]][1][action_trace[i]]
                    x = b[state_trace[i]][1][action_trace[i]]
                    rho *= y / x

                G = 0
                for i in range(tau + 1, min([tau + n, T]) + 1):
                    G += pow(GAMMA, (i - tau - 1)) * reward_trace[i - 1]

                if tau + n < T:
                    G += pow(GAMMA, n) * Q[state_trace[-1], action_trace[-1]]

                Q[state_trace[tau], action_trace[tau]] += ALPHA * rho * (G - Q[state_trace[tau], action_trace[tau]])

                if state_trace[tau] not in TERMINAL_STATES:
                    policy[state_trace[tau]] = e_greedy(env, EPSILON, Q, state_trace[tau])

            if tau == T - 1:
                break

            time_step += 1

        episode_reward = sum(reward_trace)
        episode_reward_list.append(episode_reward)
        print("EPISODE[{0}] Episode Reward: {1}".format(episode, episode_reward))

    return policy, np.asarray(episode_reward_list)


def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,   # exploring start
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )

    Q = state_action_value(env)

    # EPSILON-Greedy 정책 생성
    policy = generate_e_greedy_policy(env, EPSILON, Q)

    # 무작위 정책 생성
    b = generate_random_policy(env)

    policy, episode_reward_list = n_step_off_policy_td(env, Q, policy, b, 2)

    # print policy
    print_grid_world_policy(env, policy)


if __name__ == '__main__':
    main()