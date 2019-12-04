/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
using namespace std;

template<typename T>
GPUAdder<T>::GPUAdder (T* array_host_, int length_) {
  array_host = array_host_;
  length = length_;
  int size = length * sizeof(T);
  cudaError_t err = cudaMalloc((void**) &array_device, size);
  assert(err == 0);
  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);
}

template<typename T>
void GPUAdder<T>::increment() {
  kernel_add_one<<<64, 64>>>(array_device, length);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

template<typename T>
void GPUAdder<T>::retreive() {
  int size = length * sizeof(T);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
}

template<typename T>
void GPUAdder<T>::retreive_to (T* array_host_, int length_) {
  assert(length == length_);
  int size = length * sizeof(T);
  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

template<typename T>
GPUAdder<T>::~GPUAdder() {
  cudaFree(array_device);
}

template class GPUAdder<float>;
template class GPUAdder<double>;
template class GPUAdder<int>;
