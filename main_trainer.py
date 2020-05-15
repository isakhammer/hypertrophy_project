import numpy as np
from matplotlib import pyplot as plt
import random


max_squat = 10

V1 = 100
t1 = 30


if __name__=="__main__":

    def V_ref(t):
        V_ref = V1*t/t1
        return V_ref


    # initalize my training
    N = 10
    V = []
    t = np.linspace(0, t1, N)
    V_ref = V_ref(t)

    # input
    noise = np.random.normal(0,10,t.shape)
    V = V_ref + noise

    plt.scatter(t, V)
    plt.plot(t, V_ref)
    plt.show()







