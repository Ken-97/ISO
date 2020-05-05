import matplotlib.pyplot as plt
import numpy as np
import math


def gaussian(sigma, x, u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (
                sigma * math.sqrt(2 * math.pi))
    return y


# x = np.linspace(220, 230, 10000)
x = np.linspace(-800, 800, 10000)
plt.plot(x, gaussian(60, x, 0), "b-", linewidth=1)
plt.show()
