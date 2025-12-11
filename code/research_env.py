import numpy as np, random
TOPIC_TECH=0; TOPIC_HEALTH=1; TOPIC_GENERAL=2
LEN_SHORT=0; LEN_MEDIUM=1; LEN_LONG=2
PREV_SUCCESS=0; PREV_FAIL=1
def encode_state(topic,length_bucket,prev_fail): return topic*6 + length_bucket*2 + prev_fail
class SyntheticResearchEnv:
    def __init__(self,n_states=18,n_actions=3,sigma=0.3):
        self.n_states=n_states; self.n_actions=n_actions; self.sigma=sigma
        rng=np.random.default_rng(42)
        self.means={}
        for s in range(n_states):
            best=rng.integers(0,n_actions)
            for a in range(n_actions):
                base=rng.uniform(0,0.6)
                if a==best: base+=0.4
                self.means[(s,a)]=base
    def reset(self): return random.randint(0,self.n_states-1)
    def step(self,state,action):
        mean=self.means[(state,action)]
        r=float(np.random.normal(mean,self.sigma))
        r=max(-1,min(1,r))
        return state,r,True,{"mean":mean}
