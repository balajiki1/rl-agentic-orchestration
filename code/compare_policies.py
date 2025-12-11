import numpy as np
from q_learning import QLearningController
from ucb_bandit import UCBBandit
from research_env import SyntheticResearchEnv
env=SyntheticResearchEnv()
def baseline(ep=200):
    return np.mean([env.step(env.reset(),1)[1] for _ in range(ep)])
def q_eval(ep=200):
    Q=np.load('q_table.npy')
    agent=QLearningController(18,3); agent.Q=Q; agent.epsilon=0
    return np.mean([env.step(env.reset(),agent.select_action(env.reset()))[1] for _ in range(ep)])
def ucb_eval(ep=200):
    agent=UCBBandit(3); r=[]
    for _ in range(ep):
        s=env.reset(); a=agent.select_action()
        ns,rw,d,inf=env.step(s,a); agent.update(a,rw); r.append(rw)
    return np.mean(r)
if __name__=='__main__':
    print("Baseline:",baseline())
    print("Q-Learning:",q_eval())
    print("UCB:",ucb_eval())
