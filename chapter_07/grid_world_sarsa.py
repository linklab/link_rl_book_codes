import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from environments.gridworld import GridWorld

plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 초기 하이퍼파라미터 설정: ε=0.2, α=0.5, γ=0.98, n-스텝 = 3, 에피소드 수행 횟수 = 100
EPSILON = 0.2
ALPHA = 0.5
GAMMA = 0.98
MAX_EPISODES = 100


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


# n-스텝 SARSA 함수
def n_step_sarsa(env, Q, n):
    episode_reward = 0
    episode_reward_list = []

    policy = generate_e_greedy_policy(env, EPSILON, Q)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        action = np.random.choice(policy[state][0], p=policy[state][1])
        state_trace, action_trace, reward_trace = [state], [action], []

        # 타임 스텝
        time_step = 0

        # 이 에피소드의 길이
        T = float('inf')

        # SARSA == STATE ACTION REWARD STATE ACTION
        while True:
            if time_step < T:
                action = np.random.choice(policy[state][0], p=policy[state][1])
                next_state, reward, done, _ = env.step(action)
                reward_trace.append(reward)
                state_trace.append(next_state)

                if done:
                    T = time_step + 1
                else:
                    action = np.random.choice(policy[next_state][0], p=policy[next_state][1])
                    action_trace.append(action)

            # 갱신을 수행할 타임 스텝 결정
            tau = time_step - n + 1

            if tau >= 0:     # update_state 시작위치부터 n개를 reward_trace[]에서 가져와야 하기 때문
                # print(len(state_trace), len(action_trace), len(reward_trace)

                G = 0
                for i in range(tau + 1, min([tau + n, T]) + 1):
                    G += pow(GAMMA, (i - tau - 1)) * reward_trace[i - 1]

                if tau + n < T:
                    G += pow(GAMMA, n) * Q[state_trace[tau + n], action_trace[tau + n]]

                Q[state_trace[tau], action_trace[tau]] += ALPHA * (G - Q[state_trace[tau], action_trace[tau]])

                policy[state_trace[tau]] = e_greedy(env, EPSILON, Q, state_trace[tau])
                policy = generate_e_greedy_policy(env, EPSILON, Q)

            if tau == T - 1:
                break
            else:
                time_step += 1
            state = next_state

        episode_reward = sum(reward_trace)
        episode_reward_list.append(episode_reward)

    return np.asarray(episode_reward_list)


def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=5,
        width=5,
        start_state=(0, 0),
        terminal_states=[(4, 4)],
        transition_reward=-0.1,
        terminal_reward=1.0
    )

    step_n = [1, 2, 4, 8, 16, 32]
    data = np.zeros(shape=(len(step_n), MAX_EPISODES))

    for idx_n, n in enumerate(step_n):
        env.reset()
        Q = state_action_value(env)

        print("n={0}".format(n))

        episode_reward_list = n_step_sarsa(env, Q, n)

        avg_episode_reward_list = []
        for episode in range(MAX_EPISODES):
            avg_episode_reward_list.append(episode_reward_list[max(0, episode - 10):(episode + 1)].mean())

        data[idx_n, :] = avg_episode_reward_list

    marker = ['o', 'x', '.', 's', '*', '+', '|', '^', 'D', ' ']
    for idx_n, n in enumerate(step_n):
        plt.plot(range(0, MAX_EPISODES, 5), data[idx_n, ::5], marker=marker[idx_n], label='n = {0}'.format(step_n[idx_n]))

    plt.xlabel('에피소드')
    plt.ylabel('에피소드별 평균 리워드')
    plt.legend()

    plt.savefig('images/n_step_sarsa.png')
    plt.close()


if __name__ == '__main__':
    main()