import numpy as np

#from numba import jit

#@jit(nopython=True)
def sort_insertion(data):
    # copy data
    tmp = np.copy(data)

    for i in range(1,len(tmp)):

        # record value
        x = tmp[i]

        # shift
        j = i
        while (j > 0) and (tmp[j-1] > x):
            tmp[j] = tmp[j-1]
            j = j-1
            
        # insert
        tmp[j] = x

    # return sorted data
    return tmp


if __name__ == "__main__":
    # data size
    n = 10000
    np.random.seed(0)
    data = np.random.random(size=n)

    import time
    start = time.perf_counter()
    sorted_data = sort_insertion(data)
    end = time.perf_counter()
    print("Duree = ", end-start," secondes")

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.subplot(2,1,2)
    plt.plot(sorted_data)
    plt.show()

