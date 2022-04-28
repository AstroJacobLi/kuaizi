import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
print('Start to plot')
plt.plot(x, x)
plt.savefig('test.png')
plt.close()
print('Good')
