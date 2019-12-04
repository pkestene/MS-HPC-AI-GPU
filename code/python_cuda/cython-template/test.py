import gpuadder
import numpy as np
import numpy.testing as npt
import gpuadder

def testing_gpuadder():

    # create some input data
    arr = np.arange(128, dtype=np.float64)
    
    # create a gpuadder object
    adder = gpuadder.GPUAdder_double(arr)

    # compute on GPU
    adder.increment()

    # retrieve results
    adder.retreive_inplace()
    results = adder.retreive()

    # expected results
    answer = np.arange(1,129, dtype=np.float64)

    # compare
    npt.assert_array_equal(arr, answer)
    npt.assert_array_equal(results, answer)


if __name__ == "__main__":
    print("Testing gpuadder...")
    testing_gpuadder()
