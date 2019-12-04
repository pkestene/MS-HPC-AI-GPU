import numpy as np
cimport numpy as np

assert sizeof(int)    == sizeof(np.int32_t)
assert sizeof(float)  == sizeof(np.float32_t)
assert sizeof(double) == sizeof(np.float64_t)

cdef extern from "src/manager.hh":
    cdef cppclass GPUAdder[T]:
        GPUAdder(T*, int)
        void increment()
        void retreive()
        void retreive_to(T*, int)

cdef class GPUAdder_double:
    cdef GPUAdder[double]* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float64_t] arr):
        self.dim1 = len(arr)
        self.g = new GPUAdder[double](&arr[0], self.dim1)

    def increment(self):
        self.g.increment()

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] a = np.zeros(self.dim1, dtype=np.float64)

        self.g.retreive_to(&a[0], self.dim1)

        return a
