from chapter_10.dqn import *
from chapter_10.dueling_double_dqn import DuelingDoubleDqnAgent
from chapter_10.per_dueling_double_dqn import PrioritizedExperienceMemory, PerDuelingDoubleDqnAgent

log_dir = 'logs/advanced_dqn_pong/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class AdvancedDqnAgent(PerDuelingDoubleDqnAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__name__ = "advanced_dqn_agent"


def main():
    env = gym.make('Pong-v0')
    advanced_dqn_agent = AdvancedDqnAgent(env)
    advanced_dqn_agent.learn()
    advanced_dqn_agent.save_model()

    advanced_dqn_agent2 = AdvancedDqnAgent(env)
    advanced_dqn_agent2.load_model()
    execution(env, advanced_dqn_agent2)


if __name__ == "__main__":
    main()
    # tensorboard --logdir 'logs/advanced_dqn_agent/'