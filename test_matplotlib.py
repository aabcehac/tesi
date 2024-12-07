import matplotlib.pyplot as plt
import numpy as np

f = lambda x: 1/(1+np.exp(-x))
t = np.arange(-1, 1, 0.001)
plt.figure()
fig, ax = plt.subplots()
plt.plot(t, f(t))
fig.savefig('sigmoid.png', transparent=False, dpi=80, bbox_inches="tight")