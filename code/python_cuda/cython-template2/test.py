import gpuadder
import numpy as np
import numpy.testing as npt
import gpuadder

def test_float64():
    # create input data
    arr = np.arange(128, dtype=np.float64)
    
    # create gpuadder object
    adder = gpuadder.GPUAdder_Double(arr)

    # perform computation on GPU
    adder.increment()
    
    # retrieve results
    adder.retreive_inplace()
    results2 = adder.retreive()

    answer = np.arange(1,129)

    # check results
    npt.assert_array_equal(arr, answer)
    npt.assert_array_equal(results2, answer)

def test_int32():
    # create input data
    arr = np.arange(128, dtype=np.int32)

    # create gpuadder object
    adder = gpuadder.GPUAdder_Int(arr)

    # perform computation on GPU
    adder.increment()

    # retrieve results
    adder.retreive_inplace()
    results2 = adder.retreive()

    answer = np.arange(1,129)

    # check results
    npt.assert_array_equal(arr, answer)
    npt.assert_array_equal(results2, answer)


if __name__ == "__main__":
    print("Testing gpuadder with int32")
    test_int32()

    print("Testing gpuadder with float64")
    test_float64()

