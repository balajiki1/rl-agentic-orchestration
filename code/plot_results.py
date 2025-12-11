import numpy as np, matplotlib.pyplot as plt
q=np.load('q_learning_moving_avg.npy')
u=np.load('ucb_moving_avg.npy')
plt.plot(q,label='Q-Learning'); plt.plot(u,label='UCB')
plt.legend(); plt.title('Learning Curves'); plt.savefig('learning_curve.png')
