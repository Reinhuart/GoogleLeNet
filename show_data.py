import matplotlib.pyplot as plt
import numpy as np

def show(history):
    history = np.array(history)
    times = np.arange(len(history[:, 0]))
    plt.figure(dpi=250)
    plt.plot(times, history[:, 0], c='blue')
    plt.scatter(times, history[:, 1], c='red')
    plt.show()
    return None