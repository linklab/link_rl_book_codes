import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from environments.randomwalk import RandomWalk

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

NUM_INTERNAL_STATES = 5

# 0: 왼쪽 종료 상태 T1를 나타냄, 상태 가치는 0.0으로 변하지 않음
# 6: 오른쪽 종료 상태 T2를 나타냄, 상태 가치는 1.0으로 변하지 않음
# 1부터 5는 각각 차례로 상태 A부터 상태 E를 나타냄, 각 상태 가치는 0.5로 초기화됨
VALUES = np.zeros(NUM_INTERNAL_STATES)
VALUES[0:NUM_INTERNAL_STATES] = 0.5

# 올바른 상태 가치 값 저장
TRUE_VALUES = np.zeros(NUM_INTERNAL_STATES)
TRUE_VALUES[0:NUM_INTERNAL_STATES] = np.arange(1, 6) / 6.0

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
    batch_list = []

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

        batch_list.append([state, action, next_state, reward, done])

        state = next_state

    if batch:
        return batch_list


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def constant_alpha_monte_carlo(env, policy, values, alpha=0.1, batch=False):
    env.reset()
    batch_list = []

    done = False
    state = env.current_state
    return_ = None
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        if done:
            if next_state == 'T2':
                return_ = 1.0
            elif next_state == 'T1':
                return_ = 0.0
            else:
                raise ValueError()

        batch_list.append([state, action, next_state, return_, done])
        state = next_state

    for sample in batch_list:
        if not sample[3]:
            sample[3] = return_

    if batch:
        return batch_list
    else:
        for sample in batch_list:
            state = sample[0]
            return_ = sample[3]

            # MC 갱신
            values[state] += alpha * (return_ - values[state])


# TD(0)를 활용한 상태 가치 추정
def compute_state_values(env):
    policy = generate_initial_random_policy(env)
    episodes = [3, 10, 100]
    markers = ['o', '+', 'D']
    plt.figure()
    plt.plot(['A', 'B', 'C', 'D', 'E'], VALUES, label='초기 가치', linestyle=":")

    for i in range(len(episodes)):
        state_values = VALUES.copy()
        for _ in range(episodes[i]):
            temporal_difference(env, policy, state_values)
        plt.plot(['A', 'B', 'C', 'D', 'E'], state_values, label=str(episodes[i]) + ' 에피소드', marker=markers[i-1])

    plt.plot(['A', 'B', 'C', 'D', 'E'], TRUE_VALUES, label='올바른 가치', linestyle="--")

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
    total_runs = 200
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

        for _ in range(total_runs):
            errors = []
            state_values = np.copy(VALUES)
            for i in range(episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUES - state_values, 2)) / NUM_INTERNAL_STATES))
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
def batch_updating(env, method, episodes, alpha=0.001):
    policy = generate_initial_random_policy(env)

    # episodes 횟수의 독립적인 수행 결과 모음
    total_errors = np.zeros(episodes)

    total_runs = 10
    for _ in range(total_runs):
        state_values = np.copy(VALUES)
        errors = []

        for _ in range(episodes):
            if method == 'TD(0)':
                batch_list = temporal_difference(env, policy, None, batch=True)
            elif method == 'MC':
                batch_list = constant_alpha_monte_carlo(env, policy, None, batch=True)
            else:
                raise ValueError()

            for i in range(100):
                updates = np.zeros(NUM_INTERNAL_STATES)
                counts = np.zeros(NUM_INTERNAL_STATES)
                for sample in batch_list:
                    if method == 'TD(0)':
                        state = sample[0]
                        next_state = sample[2]
                        reward = sample[3]
                        done = sample[4]

                        if done:
                            updates[state] += reward - state_values[state]
                            counts[state] += 1
                        else:
                            updates[state] += reward + state_values[next_state] - state_values[state]
                            counts[state] += 1
                    elif method == 'MC':
                        state = sample[0]
                        return_ = sample[3]

                        updates[state] += return_ - state_values[state]
                        counts[state] += 1
                    else:
                        raise ValueError()

                for state in range(NUM_INTERNAL_STATES):
                    if counts[state]:
                        updates[state] = updates[state] / counts[state]

                    state_values[state] += alpha * updates[state]

            # RMS 에러 계산
            errors.append(np.sqrt(np.sum(np.power(TRUE_VALUES - state_values, 2)) / NUM_INTERNAL_STATES))
        total_errors += np.asarray(errors)

    total_errors /= total_runs
    return total_errors, state_values


def batch_updating_execution_alpha_1(env):
    episodes = 100 + 1

    td_errors, td_state_values = batch_updating(env, 'TD(0)', episodes, alpha=0.001)
    mc_errors, mc_state_values = batch_updating(env, 'MC', episodes, alpha=0.001)

    with np.printoptions(precision=2, suppress=True):
        print('TD(0)', 'alpha=0.001', td_state_values)
        print('MC', 'alpha=0.001', mc_state_values)

    plt.plot(td_errors, label='TD(0)')
    plt.plot(mc_errors, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating_alpha_1.png')
    plt.close()


def batch_updating_execution_alpha_2(env):
    episodes = 100 + 1

    td_errors, td_state_values = batch_updating(env, 'TD(0)', episodes, alpha=0.002)
    mc_errors, mc_state_values = batch_updating(env, 'MC', episodes, alpha=0.002)

    with np.printoptions(precision=2, suppress=True):
        print('TD(0)', 'alpha=0.002', td_state_values)
        print('MC', 'alpha=0.002', mc_state_values)

    plt.plot(td_errors, label='TD(0)')
    plt.plot(mc_errors, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating_alpha_2.png')
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
    batch_updating_execution_alpha_1(env)
    batch_updating_execution_alpha_2(env)


if __name__ == '__main__':
    main()

