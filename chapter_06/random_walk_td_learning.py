import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.randomwalk import RandomWalk

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

NUM_INTERNAL_STATES = 5

# 0: 왼쪽 종료 상태 T1를 나타냄, 상태 가치는 0.0으로 변하지 않음
# 6: 오른쪽 종료 상태 T2를 나타냄, 상태 가치는 1.0으로 변하지 않음
# 1부터 5는 각각 차례로 상태 A부터 상태 E를 나타냄, 각 상태 가치는 0.5로 초기화됨
VALUES = np.zeros(NUM_INTERNAL_STATES)

# 올바른 상태 가치 값 저장
TRUE_VALUE = np.zeros(NUM_INTERNAL_STATES)
TRUE_VALUE[0:5] = np.arange(1, 6) / 6.0

# 종료 상태를 제외한 임의의 상태에서 동일한 확률로 왼쪽 이동 또는 오른쪽 이동
LEFT_ACTION = 0
RIGHT_ACTION = 1

# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for state in env.observation_space.STATES:
        actions = []
        prob = []
        for action in range(env.action_space.num_actions):
            actions.append(action)
            prob.append(0.5)
        policy[state] = (actions, prob)

    return policy


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def temporal_difference(env, policy, state_values, alpha=0.1, batch=False):
    env.reset()
    trajectory = [env.current_state]
    rewards = [0]
    done = False
    state = env.current_state
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        # TD 갱신 수행
        if not batch:
            if done:
                state_values[state] += alpha * (reward - state_values[state])
            else:
                state_values[state] += alpha * (reward + state_values[next_state] - state_values[state])

        state = next_state

        trajectory.append(state)
        rewards.append(reward)

    if batch:
        return trajectory, rewards


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def constant_alpha_monte_carlo(env, policy, values, alpha=0.1, batch=False):
    env.reset()
    trajectory = [env.current_state]
    done = False
    state = env.current_state

    returns = 0.0
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        trajectory.append(state)

        if done:
            if next_state == 'T2':
                returns = 1.0

        state = next_state

    if batch:
        return trajectory, [returns] * (len(trajectory) - 1)
    else:
        for state_ in trajectory[:-1]:
            # MC 갱신
            values[state_] += alpha * (returns - values[state_])


# TD(0)를 활용한 상태 가치 추정
def compute_state_values(env):
    policy = generate_initial_random_policy(env)
    episodes = [3, 10, 100]
    markers = ['o', '+', 'D']
    plt.figure()
    state_values = VALUES
    plt.plot(['A', 'B', 'C', 'D', 'E'], state_values, label='초기 가치', linestyle=":")

    for i in range(len(episodes)):
        state_values = VALUES.copy()
        for _ in range(episodes[i]):
            temporal_difference(env, policy, state_values)
        plt.plot(['A', 'B', 'C', 'D', 'E'], state_values, label=str(episodes[i]) + ' 에피소드', marker=markers[i-1])

    plt.plot(['A', 'B', 'C', 'D', 'E'], TRUE_VALUE, label='올바른 가치', linestyle="--")

    plt.xlabel('상태')
    plt.ylabel('추정 가치')
    plt.legend()
    plt.savefig('images/random_walk_td_prediction.png')
    plt.close()


# TD(0)와 상스-alpha MC의 상태 가치 예측 성능 비교
def rms_errors(env):
    policy = generate_initial_random_policy(env)

    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05, 0.025]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    total_runs = 20
    episodes = 200 + 1
    plt.figure()

    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD(0)'
            linestyle = '-'
        else:
            method = 'MC'
            linestyle = '-.'

        for _ in tqdm(range(total_runs)):
            errors = []
            state_values = np.copy(VALUES)
            for i in range(episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - state_values, 2)) / 7.0))
                if method == 'TD(0)':
                    temporal_difference(env, policy, state_values, alpha=alpha)
                else:
                    constant_alpha_monte_carlo(env, policy, state_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= total_runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = {0:.2f}'.format(alpha))

    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()
    plt.savefig('images/random_walk_rms_errors_comparison.png')
    plt.close()


# @method: 'TD(0)' 또는 'MC'
def batch_updating(method, episodes, alpha=0.001):
    # episodes 횟수의 독립적인 수행 결과 모음
    total_errors = np.zeros(episodes)

    total_runs = 100
    for _ in tqdm(range(total_runs)):
        V = np.copy(VALUES)
        errors = []

        trajectories = []
        rewards = []
        for _ in range(episodes):
            if method == 'TD(0)':
                #
                trajectory_, rewards_ = temporal_difference(V, batch=True)
            else:
                trajectory_, rewards_ = constant_alpha_monte_carlo(V, batch=True)

            trajectories.append(trajectory_)
            rewards.append(rewards_)

            while True:
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(len(trajectory_) - 1):
                        if method == 'TD(0)':
                            updates[trajectory_[i]] += rewards_[i] + V[trajectory_[i + 1]] - V[trajectory_[i]]
                        else:
                            returns = rewards_[i]
                            updates[trajectory_[i]] += returns - V[trajectory_[i]]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # 배치 업데이트
                V += updates
            # RMS 에러 계산
            errors.append(np.sqrt(np.sum(np.power(V - TRUE_VALUE, 2)) / 5.0))
        total_errors += np.asarray(errors)
    total_errors /= total_runs
    return total_errors


def batch_updating_execution_episode():
    episodes = 100 + 1
    td_erros = batch_updating('TD(0)', episodes)
    mc_erros = batch_updating('MC', episodes)

    plt.plot(td_erros, label='TD(0)')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating.png')
    plt.close()


def batch_updating_execution_alpha():
    episodes = 100 + 1
    td_erros = batch_updating('TD(0)', episodes, alpha=0.002)
    mc_erros = batch_updating('MC', episodes, alpha=0.002)

    plt.plot(td_erros, label='TD(0)')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating_alpha.png')
    plt.close()


def main():
    env = RandomWalk(
        num_internal_states=NUM_INTERNAL_STATES,
        transition_reward=0.0,
        left_terminal_reward=0.0,
        right_terminal_reward=1.0
    )
    compute_state_values(env)
    rms_errors(env)
    # batch_updating_execution_episode()
    # batch_updating_execution_alpha()


if __name__ == '__main__':
    main()

