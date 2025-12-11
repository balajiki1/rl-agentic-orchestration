import numpy as np
from q_learning import QLearningController
from research_env import SyntheticResearchEnv
def train_q_learning(episodes=1000):
    env=SyntheticResearchEnv()
    agent=QLearningController(18,3)
    rewards=[]; mov=[]
    for ep in range(episodes):
        s=env.reset(); a=agent.select_action(s)
        ns,r,d,info=env.step(s,a)
        agent.update(s,a,r,ns,d); agent.decay_epsilon()
        rewards.append(r)
        mov.append(np.mean(rewards[-50:]))
    np.save('q_learning_rewards.npy',rewards)
    np.save('q_learning_moving_avg.npy',mov)
    agent.save('q_table.npy')
if __name__=='__main__': train_q_learning()
