from research_env import encode_state,TOPIC_TECH,TOPIC_HEALTH,TOPIC_GENERAL,LEN_SHORT,LEN_MEDIUM,LEN_LONG,PREV_SUCCESS,PREV_FAIL
class OrchestratorStub:
    def __init__(self): self.last_success=PREV_SUCCESS
    def encode_state_for_query(self,q):
        ql=q.lower()
        topic=TOPIC_TECH if any(x in ql for x in ['api','react','python','ai']) else TOPIC_HEALTH if any(x in ql for x in ['health','diabetes','glp']) else TOPIC_GENERAL
        n=len(q.split()); length = LEN_SHORT if n<=6 else LEN_MEDIUM if n<=15 else LEN_LONG
        return encode_state(topic,length,self.last_success)
    def run_pipeline(self,a,q):
        if a==0: return {"answer":"fast","tools":1,"quality":0.4}
        if a==1: return {"answer":"deep","tools":3,"quality":0.8}
        if a==2: return {"answer":"rag","tools":2,"quality":0.7}
        return {"answer":"fallback","tools":1,"quality":0.0}
    def reward(self,quality,tools,w1=1.0,w2=0.2): return w1*quality - w2*tools
