#! /usr/bin/env make

# you have set CUDA_PATH env variable either manually
# or using "module load cuda"
CUDA_INC_PATH=${CUDA_PATH}/include
CUDA_LIB_PATH=${CUDA_PATH}/lib64
CUDA_ARCH ?= sm_80


CC=nvcc
CFLAGS= -O3 -I$(CUDA_INC_PATH) --ptxas-options=-v -arch=${CUDA_ARCH} -Xcompiler -fopenmp
LDFLAGS=-L$(CUDA_LIB_PATH) -lcudart -lcublas -lgomp

all: saxpy

saxpy:
	$(CC) $(CFLAGS) $(LDFLAGS) my_cuda_utils.cu SimpleTimer.cpp  OpenMPTimer.cpp saxpy.cu -o saxpy

clean:
	\rm -f *.o
	\rm -f saxpy
	\rm -f *~
	\rm -f *.linkinfo *.log









