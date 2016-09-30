import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('log.txt')

plt.figure()
plt.plot(data)
plt.show(block=True)
