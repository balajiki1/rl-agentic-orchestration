import numpy as np, random
class QLearningController:
    def __init__(self,n_states,n_actions,alpha=0.1,gamma=0.95,epsilon=0.2,epsilon_min=0.05,epsilon_decay=0.995):
        self.n_states=n_states; self.n_actions=n_actions
        self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay
        self.Q=np.zeros((n_states,n_actions))
    def select_action(self,state):
        return random.randint(0,self.n_actions-1) if random.random()<self.epsilon else int(np.argmax(self.Q[state]))
    def update(self,state,action,reward,next_state,done):
        best_next=0 if done else np.max(self.Q[next_state])
        td_target=reward+(0 if done else self.gamma*best_next)
        self.Q[state,action]+=self.alpha*(td_target-self.Q[state,action])
    def decay_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
    def save(self,p): np.save(p,self.Q)
    def load(self,p): self.Q=np.load(p)
