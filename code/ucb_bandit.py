import math
class UCBBandit:
    def __init__(self,n_actions,c=2.0):
        self.n_actions=n_actions; self.c=c
        self.counts=[0]*n_actions; self.values=[0.0]*n_actions
        self.total_pulls=0
    def select_action(self):
        for a in range(self.n_actions):
            if self.counts[a]==0: return a
        self.total_pulls+=1
        vals=[ self.values[a] + self.c*math.sqrt(math.log(self.total_pulls)/self.counts[a]) for a in range(self.n_actions)]
        return int(max(range(self.n_actions),key=lambda x:vals[x]))
    def update(self,a,r):
        self.counts[a]+=1; n=self.counts[a]
        self.values[a]+= (r-self.values[a])/n
