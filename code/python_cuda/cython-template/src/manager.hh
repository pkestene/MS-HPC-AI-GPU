template<typename T>
class GPUAdder {
  // pointer to the GPU memory where the array is stored
  T* array_device;
  // pointer to the CPU memory where the array is stored
  T* array_host;
  // length of the array (number of elements)
  int length;

public:
  GPUAdder(T* INPLACE_ARRAY1, int DIM1); // constructor (copies to GPU)

  ~GPUAdder(); // destructor

  void increment(); // does operation inplace on the GPU

  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void retreive_to (T* INPLACE_ARRAY1, int DIM1);


};
