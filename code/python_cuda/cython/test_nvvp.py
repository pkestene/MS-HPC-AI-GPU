#!/usr/bin/env python

import gpuadder
import numpy as np
import numpy.testing as npt
import gpuadder

def test():
    arr = np.array([1,2,5,2,-3], dtype=np.int32)
    print("input array :")
    print(arr)

    adder = gpuadder.GPUAdder(arr)
    adder.increment()
    
    adder.retreive_inplace()
    results2 = adder.retreive()

    print("output array:")
    print(arr)
    
    npt.assert_array_equal(arr, [2,3,6,3,-2])
    npt.assert_array_equal(results2, [2,3,6,3,-2])

    
    
test()
