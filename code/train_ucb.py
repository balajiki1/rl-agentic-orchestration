import numpy as np
from ucb_bandit import UCBBandit
from research_env import SyntheticResearchEnv
def train_ucb(episodes=1000):
    env=SyntheticResearchEnv(); agent=UCBBandit(3)
    rewards=[]; mov=[]
    for ep in range(episodes):
        s=env.reset(); a=agent.select_action()
        ns,r,d,info=env.step(s,a)
        agent.update(a,r)
        rewards.append(r); mov.append(np.mean(rewards[-50:]))
    np.save('ucb_rewards.npy',rewards)
    np.save('ucb_moving_avg.npy',mov)
if __name__=='__main__': train_ucb()
