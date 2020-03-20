# 사용 패키지 임포트
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from environments.gridworld import GridWorld

plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

STEP_N_MAX = 9


# 행동-가치 함수 생성
def state_action_value(env):
    q = dict()
    for state in env.observation_space.STATES:
        for action in env.observation_space.ACTIONS:
            q[(state, action)] = np.random.normal()
    return q


# 탐욕적 정책을 생성하는 함수
def generate_greedy_policy(env, Q):
    policy = dict()
    for state in env.observation_space.STATES:
        actions = []
        q_values = []
        prob = []

        for action in env.observation_space.ACTIONS:
            actions.append(action)
            q_values.append(Q[state, action])

        for i in range(len(q_values)):
            if i == np.argmax(q_values):
                prob.append(1)
            else:
                prob.append(0)

        policy[state] = (actions, prob)
    return policy


# ε-탐욕적 정책의 확률 계산 함수
def e_greedy(env, e, q, state):
    action_values = []
    prob = []
    for action in env.observation_space.ACTIONS:
        action_values.append(q[(state, action)])

    for i in range(len(action_values)):
        if i == np.argmax(action_values):
            prob.append(1 - e + e/len(action_values))
        else:
            prob.append(e/len(action_values))
    return env.observation_space.ACTIONS, prob


# ε-탐욕적 정책 생성 함수
def generate_e_greedy_policy(env, e, Q):
    policy = dict()
    for state in env.observation_space.STATES:
        policy[state] = e_greedy(env, e, Q, state)
    return policy


# n-스텝 SARSA 함수
# 초기 하이퍼파라미터 설정: ε=0.3, α=0.5, γ=0.98, n-스텝 = 3, 반복 수행 횟수 = 100
def n_step_sarsa(env, epsilon=0.3, alpha=0.5, gamma=0.98, n=3, num_iter=100, learn_policy=True):
    Q = state_action_value(env)
    policy = generate_e_greedy_policy(env, epsilon, Q)

    cumulative_reward = 0

    for _ in range(num_iter):
        state = env.reset()
        action = np.random.choice(policy[state][0], p=policy[state][1])
        state_trace, action_trace, reward_trace = [state], [action], []
        t, T = 0, 10000

        # SARSA == STATE ACTION REWARD STATE ACTION
        while True:
            if t < T:
                next_state, reward, done, _ = env.step(action)
                reward_trace.append(reward)
                state_trace.append(next_state)

                if done:
                    T = t + 1
                    cumulative_reward += sum(reward_trace)      # episode 누적 reward
                else:
                    action = np.random.choice(policy[next_state][0], p=policy[next_state][1])
                    action_trace.append(action)

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min([tau + n, T]) + 1):
                    G += (gamma ** (i - tau - 1)) * reward_trace[i - 1]

                if tau + n < T:
                    G += (gamma ** n) * Q[state_trace[tau + n], action_trace[tau + n]]

                Q[state_trace[tau], action_trace[tau]] += alpha * (G - Q[state_trace[tau], action_trace[tau]])

                if learn_policy:
                    policy[state_trace[tau]] = e_greedy(env, epsilon, Q, state_trace[tau])

            if tau == (T - 1):
                break
            t += 1

    return policy, Q, cumulative_reward/num_iter


if __name__ == '__main__':
    average_reward_lst = []

    # 그리드 월드 환경 객체 생성
    env = GridWorld(transition_reward=-0.1)

    step_n = np.power(2, np.arange(0, 9))
    alphas = np.arange(0.1, 1.1, 0.1)

    average_rewards = np.zeros((len(step_n), len(alphas)))
    for n_idx, n in enumerate(step_n):
        for alpha_idx, alpha in enumerate(alphas):
            for _ in range(100):
                policy, Q, average_reward = n_step_sarsa(env, epsilon=0.2, alpha=alpha, gamma=0.98, n=n, num_iter=10)
                average_reward_lst.append(average_reward)

            average_rewards[n_idx, alpha_idx] = sum(average_reward_lst)/len(average_reward_lst)
            print("step_n:", n, " alphas:", alpha)
            # print(policy)
            # print(Q)
            print("average_reward:", average_rewards[n_idx, alpha_idx])

    marker = ['o', 'x', '.', 's', '*', '+', '|', '^', 'D', ' ']
    for i in range(0, len(step_n)):
        plt.plot(alphas, average_rewards[i, :], marker=marker[i], label='n = %d' % (step_n[i]))

    plt.xlabel('스텝 사이즈(alpha)')
    plt.ylabel('episode 평균 reward')
    plt.ylim([-10, -5])
    plt.legend()

    plt.savefig('images/n_step_sarsa_for_grid_world.png')
    plt.close()
