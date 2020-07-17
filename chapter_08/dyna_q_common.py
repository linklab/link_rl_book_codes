import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
import heapq
from copy import deepcopy
import random
from queue import PriorityQueue

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


class DynaQParams:
    # 감가율
    gamma = 0.95

    # 탐색(exploration) 확률
    epsilon = 0.1

    # 스텝 사이즈
    alpha = 0.1

    # 경과 시간에 대한 가중치
    time_weight = 0

    # 계획에서의 수행 스텝 수
    max_planning_steps = 5

    # 총 수행 횟수 (성능에 대한 평균을 구하기 위함)
    runs = 10

    # 알고리즘 이름
    methods = ['Dyna-Q', 'Dyna-Q+']

    # 우선순위 코에 대한 임계값
    theta = 0


# epsilon-탐욕 알고리즘에 의해 행동 선택
def choose_action(state, q_value, dyna_maze, dyna_params):
    rand_prob = np.random.random()
    if rand_prob < dyna_params.epsilon:
        return np.random.choice(dyna_maze.ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


# Dyna-Q의 계획 과정에서 사용하는 간단한 모델
class SimpleModel:
    def __init__(self):
        self.model = dict()

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if state not in self.model.keys():
            self.model[state] = dict()

        self.model[state][action] = [reward, next_state]

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환험
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state = self.model[state][action]
        return state, action, reward, next_state


# Dyna-Q+를 위한 시간 기반 모델
class TimeModel:
    # @all_actions: 모든 행동 집합 [UP, DOWN, LEFT, RIGHT]
    # @time_weight: 샘플링된 보상에 대하여 경과 시간에 대한 가중치
    def __init__(self, all_actions, time_weight=0.0001):
        self.model = dict()

        # 전체 타임 스텝 진행 기록
        self.time = 0
        self.time_weight = time_weight
        self.all_actions = all_actions

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        if state not in self.model.keys():
            self.model[state] = dict()

            # 임의의 상태에 대하여 선택되어지지 않은 행동들에 대해서도
            # 계획 단계에 고려되도록 유도
            for action_ in self.all_actions:
                if action_ != action:
                    # 그러한 행동들은 행동 수행 이후에 상태가 변동 없으며
                    # 보상은 0로 설정
                    self.model[state][action_] = [0, state, 0]

        self.model[state][action] = [reward, next_state, self.time]
        self.time += 1

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state, time = self.model[state][action]

        # 모델에 기록된 방문 시각과 현재 시각 사이의 차가 클 수록,
        # 즉 오래전에 기록된 모델 정보일 수록 reward가 커지도록 조정함.
        # 따라서, 오래전에 경험해보고 최근에 경험해보지 못한
        # 상태-행동 쌍이 좀 더 자주 발현되도록 유도함
        reward += self.time_weight * np.sqrt(self.time - time)

        return state, action, reward, next_state


# 우선순위 스위핑을 위한 우선순위 큐를 포함한 환경 모델
class PriorityModel(SimpleModel):
    def __init__(self):
        SimpleModel.__init__(self)
        # 우선순위 큐
        self.priority_queue = PriorityQueue()
        # 임의의 상태에 대한 선행 상태를 저장
        self.predecessors = dict()

    # (@state-@action) 쌍을 우선순위 큐에 삽입. 이 때 우선순위를 @priority로서 제공
    def insert(self, priority, state, action):
        # 기존의 우선순위 큐는 minimum heap이기 때문에, 우선순위의 부호를 변경하여 삽입
        self.priority_queue.put((-priority, (state, action)))

    def is_empty(self):
        return self.priority_queue.empty()

    # 우선순위 큐에서 첫 번째 아이템을 가져옴
    def sample(self):
        state, action = self.priority_queue.get()[1]
        reward, next_state = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return state, action, reward, next_state

    # 모델에 이전의 경험을 저장함
    def store(self, state, action, reward, next_state):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        SimpleModel.store(self, state, action, reward, next_state)

        if next_state not in self.predecessors.keys():
            self.predecessors[next_state] = set()

        self.predecessors[next_state].add((state, action))

    # 임의의 상태에 대한 이전 상태를 모두 리턴함
    def predecessor(self, state):
        if state not in self.predecessors.keys():
            return []

        predecessors = []
        for state_pre, action_pre in self.predecessors[state]:
            predecessors.append(
                [state_pre, action_pre, self.model[state_pre][action_pre][0]]
            )
        return predecessors


# Dyna-Q 알고리즘의 각 에피소드 별 학습
# @q_value: 행동 가치 테이블, dyna_q 함수 수행 후 값이 갱신 됨
# @model: 계획시 사용할 모델
# @dyna_maze: 미로 환경
def dyna_q(q_value, model, dyna_maze, dyna_params):
    state = dyna_maze.START_STATE
    steps = 0
    rewards = 0.0
    while state not in dyna_maze.GOAL_STATES:
        # 타임 스텝 기록
        steps += 1
        # 행동 얻어오기
        action = choose_action(state, q_value, dyna_maze, dyna_params)
        # 행동 수행후
        reward, next_state = dyna_maze.step(state, action)

        # Q-러닝 갱신
        target = reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += dyna_params.alpha * (target - q_value[state[0], state[1], action])

        # 경험 샘플을 모델에 저장 (모델 구성)
        model.store(state, action, reward, next_state)

        # 모델로 부터 샘플 얻어오면서 Q-계획 반복 수행
        for t in range(0, dyna_params.max_planning_steps):
            state_, action_, reward_, next_state_ = model.sample()
            target = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :])
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * (target - q_value[state_[0], state_[1], action_])

        state = next_state
        rewards += reward

        # 최대 스텝 체크
        if steps > dyna_maze.max_steps:
            break

    return steps, rewards


def dyna_q_2(dyna_maze, dyna_params, original_obstacles, new_obstacles, obstacle_switch_time):
    # 누적 보상을 기록하는 자료 구조
    cumulative_rewards = np.zeros((dyna_params.runs, len(dyna_params.methods), dyna_maze.max_steps))

    print_changing_maze = False

    for run in range(dyna_params.runs):
        # 두 개의 모델 설정
        models = [SimpleModel(), TimeModel(dyna_maze.ACTIONS, time_weight=dyna_params.time_weight)]

        # 두 개의 모델별 행동 가치 초기화
        q_values = [np.zeros(dyna_maze.q_size), np.zeros(dyna_maze.q_size)]

        for method_idx in range(len(dyna_params.methods)):
            # print('run:', run, DynaQParams.methods[method_idx])
            # set old obstacles for the maze
            dyna_maze.obstacles = original_obstacles

            steps = 0
            last_steps = steps

            while steps < dyna_maze.max_steps:
                # play for an episode
                steps_, rewards = dyna_q(
                    q_values[method_idx], models[method_idx], dyna_maze, dyna_params
                )
                steps += steps_

                # update cumulative rewards
                cumulative_rewards[run, method_idx, last_steps: steps] = cumulative_rewards[run, method_idx, last_steps]
                cumulative_rewards[run, method_idx, min(steps, dyna_maze.max_steps - 1)] = cumulative_rewards[run, method_idx, last_steps] + rewards
                last_steps = steps

                if steps > obstacle_switch_time:
                    # 장애물 구성 변경
                    dyna_maze.obstacles = new_obstacles
                    if not print_changing_maze:
                        print(dyna_maze)
                        print_changing_maze = True

                # print(DynaQParams.methods[method_idx], steps, dyna_maze.obstacles)

    # 전체 수행(runs)에 대하여 평균 산출
    rewards = cumulative_rewards.mean(axis=0)

    return rewards


# 하나의 에피소드에 대해서 우선순위 스위핑 알고리즘 수행
# @q_value: 행동 가치 테이블, dyna_q 함수 수행 후 값이 갱신 됨
# @model: 계획시 사용되는 환경 모델
# @dyna_maze: 미로 환경
# @DynaQParams: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, dyna_maze, dyna_params):
    state = dyna_maze.START_STATE

    # 에피소드 내에서의 스텝 수
    steps = 0

    # 계획 과정에서의 역갱신 횟수
    backups = 0

    while state not in dyna_maze.GOAL_STATES:
        steps += 1

        # 행동 추출
        action = choose_action(state, q_value, dyna_maze, dyna_params)

        # 행동 수행 및 보상과 다음 상태 얻어오기
        reward, next_state = dyna_maze.step(state, action)

        # 모델에 경험 저장
        model.store(state, action, reward, next_state)

        # 현재 (상태, 행동) 쌍에 대한 우선순위 계산
        priority = np.abs(
            reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], action]
        )

        # 모델에 우선순위 저장
        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # start planning
        planning_steps = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_steps < dyna_params.max_planning_steps and not model.is_empty():
            # get a sample with highest priority from the model
            state_, action_, reward_, next_state_ = model.sample()

            # update the state action value for the sample
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - q_value[state_[0], state_[1], action_]

            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(
                    reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) - q_value[state_pre[0], state_pre[1], action_pre]
                )

                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)

            planning_steps += 1

        state = next_state

        # update the # of backups
        backups += planning_steps + 1

    return backups


# 행동 가치함수를 표 형태로 그리는 함수
def draw_q_value_image(dyna_maze, q_value, run, planning_steps, episode):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = dyna_maze.MAZE_HEIGHT, dyna_maze.MAZE_WIDTH
    width, height = 1.0 / num_cols, 1.0 / num_rows

    for i in range(dyna_maze.MAZE_HEIGHT):
        for j in range(dyna_maze.MAZE_WIDTH):
            if np.sum(q_value[i][j]) == 0.0:
                symbol = " "
            else:
                action_idx = np.argmax(q_value[i][j])
                symbol = dyna_maze.ACTION_SYMBOLS[action_idx]
            table.add_cell(i, j, width, height, text=symbol, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(dyna_maze.MAZE_HEIGHT):
        table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')

    for j in range(dyna_maze.MAZE_WIDTH):
        table.add_cell(-1, j, width, height/2, text=j, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    axis.add_table(table)
    plt.savefig('images/maze_action_values_{0}_{1}_{2}.png'.format(run, planning_steps, episode))
    plt.close()
