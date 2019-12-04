import gpuadder
import numpy as np
import numpy.testing as npt
import gpuadder

def testing_gpuadder():

    # create some input data
    arr = np.arange(128, dtype=np.int32)
    
    # create a gpuadder object
    adder = gpuadder.GPUAdder(arr)

    # compute on GPU
    adder.increment()

    # retrieve results inplace (on CPU)
    adder.retreive_inplace()

    # retrive results out of place (on CPU)
    results = adder.retreive()

    # true answer is an array containing data shifted by one
    answer = np.arange(1,129, dtype=np.int32)

    # compare true answer with retrieve data from GPU
    npt.assert_array_equal(arr, answer)
    npt.assert_array_equal(results, answer)


if __name__ == "__main__":
    print("Testing gpuadder...")
    testing_gpuadder()
