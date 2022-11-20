import numpy as np
import matplotlib.pyplot as plt


def N5K3():
    x = [0, 1, 2, 3, 4]
    my_cora_full = [0.6896, 0.688, 0.6655999999999999, 0.6300000000000001, 0.6200000000000001]
    cora_full = [0.6496000000000001, 0.6748000000000001, 0.6456000000000001, 0.6487999999999999, 0.6652]
    l1 = plt.plot(x, my_cora_full, 'r--', label='our results')
    l2 = plt.plot(x, cora_full, 'g--', label='their results')
    plt.plot(x, my_cora_full, 'ro-', x, cora_full, 'g+-')
    plt.title('compare the result')
    plt.xlabel('row')
    plt.ylabel('column')
    plt.legend()
    plt.show()

N5K3()