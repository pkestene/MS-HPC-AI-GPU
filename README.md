# MS-HPC-AI-GPU

Resources pour le cours d'introduction à la programmation des GPUs du [mastère spécialisé HPC-AI](https://www.hpc-ai.mines-paristech.fr/)

## biblio

### Scientific Programming and Computer Architecture

- [Scientific Programming and Computer Architecture](https://github.com/divakarvi/bk-spca) book by Divakar Viswanath
- https://github.com/divakarvi/bk-spca : source associated to the book
- [Definition of latency oriented architecture](https://en.wikipedia.org/wiki/Latency_oriented_processor_architecture)
- [Seven Dwarfs of HPC](https://moodle.rrze.uni-erlangen.de/course/view.php?id=113&lang=en)
- [CSCI-5576/4576: High Performance Scientific Computing](https://github.com/cucs-hpsc/hpsc-class)
- Mark Horowitz talk at ISSCC_2014: [Computing's energy problem](http://eecs.oregonstate.edu/research/vlsi/teaching/ECE471_WIN15/mark_horowitz_ISSCC_2014.pdf)
- [Introduction to High-Performance Scientific Computing, book and slides by Victor Eijkhout](https://pages.tacc.utexas.edu/~eijkhout/istc/istc.html)
- [San Diego Summer institute](https://github.com/sdsc/sdsc-summer-institute-2019)
- [Finnish CSC summer school](https://github.com/csc-training/summerschool)
- [Computational Physics book by K. N. Anagnostopoulos](http://www.physics.ntua.gr/~konstant/ComputationalPhysics/C++/Book/ComputationalPhysicsKNA2ndEd_nocover.pdf)
- [Modern computer architecture slides](https://moodle.rrze.uni-erlangen.de/course/view.php?id=274), see e.g. slides [Intro_Architecture.pdf](https://moodle.rrze.uni-erlangen.de/pluginfile.php/12916/mod_resource/content/9/01_Intro-Architecture.pdf)
- [Structure and Interpretation of Computer Programs](https://library.oapen.org/handle/20.500.12657/26092)

### CUDA / GPU training

- [NVIDIA's latest CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Julich training on CUDA](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Services/Documentation/presentations/presentation-cuda_table.html?nn=362392)
- [Oxford training on CUDA](https://people.maths.ox.ac.uk/gilesm/cuda/)
- [Swiss CSCS summer school](https://github.com/eth-cscs/SummerSchool2019.git)
- [Amina Guermouche (Telecom Paris)](http://www-inf.telecom-sudparis.eu/COURS/CSC5001/new_site/Supports/Cours/GPU/csc5001-cuda.pdf)
- [EPCC, Univ Edinburgh, GPU training](https://github.com/EPCCed/archer-gpu-course)
- [ARCHER GPU course](https://github.com/EPCCed/archer-gpu-course)
- [Univ Luxembourg HPC](https://www.hpcwire.com/2019/11/14/at-sc19-what-is-urgenthpc-and-why-is-it-needed/)
- [SC19 Introduction to GPU programming with CUDA](http://icl.utk.edu/~mgates3/gpu-tutorial/)
- https://codingbyexample.com/category/cuda/
- http://turing.une.edu.au/~cosc330/lectures/display_notes.php?lecture=18
- https://www.nersc.gov/users/training/gpus-for-science/
- https://dl.acm.org/citation.cfm?id=3318192
- git@bitbucket.org:hwuligans/gputeachingkit-labs.git
- http://syllabus.gputeachingkit.com/
- [udemy/cuda-programming-masterclass](https://www.udemy.com/cuda-programming-masterclass/)
- SDL2 Graphics User Interface : https://github.com/rogerallen/smandelbrotr
- [mgbench](https://github.com/tbennun/mgbench) : a multi-GPU benchmark 
- performance analysis : [parallelforall blog on Nsight](https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+nvidia%2Fparallelforall+%28NVIDIA+Parallel+Forall+Blog%29)
- misc : [convert CUDA to portable C++ for AMD GPU](https://github.com/ROCm-Developer-Tools/HIP)
- [List of Nvidia GPUs](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units)
- https://github.com/ashokyannam/GPU_Acceleration_Using_CUDA_C_CPP
- https://github.com/karlrupp/cpu-gpu-mic-comparison
- https://perso.centrale-marseille.fr/~gchiavassa/visible/HPC/01%20-%20GR%20%20Intro%20to%20GPU%20programming%20V2%20OpenACC%20.pdf

### CUDA / performance analysis

- https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/
- https://www.olcf.ornl.gov/wp-content/uploads/2019/08/NVIDIA-Profilers.pdf
- http://on-demand.gputechconf.com/gtc/2017/presentation/s7445-jakob-progsch-what-the-profiler-is-telling-you.pdf
- monitoring performance : https://github.com/NERSC/timemory
- [roofline model](https://www.nersc.gov/assets/Uploads/Talk-GTC2019-Roofline.pdf)

### Other CUDA resources

- [C++ wrapper library](https://github.com/eyalroz/cuda-api-wrappers)
- [template CMake project for CUDA](https://github.com/pkestene/cuda-proj-tmpl)

### CUDA / python

- [Numba](http://numba.pydata.org/) // [recommended numba tutorial for GPU programming](https://github.com/ContinuumIO/gtc2019-numba)
- [CuPy](https://cupy.chainer.org/)
- [pycuda](https://documen.tician.de/pycuda/)
- [python / C++ CUDA interface (SWIG and Cython)](https://github.com/pkestene/npcuda-example)
- [python / C++ CUDA interface with pybind11](https://github.com/pkestene/pybind11-cuda)
- [legate](https://legion.stanford.edu/pdfs/legate-preprint.pdf)
- [PythonHPC](https://github.com/eth-cscs/PythonHPC)
- [HPC Python video's](https://www.cscs.ch/publications/tutorials/2018/high-performance-computing-with-python/)
- [Hands-On GPU Programming with Python and CUDA](https://www.oreilly.com/library/view/hands-on-gpu-programming/9781788993913/) and [examples](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/tree/9e3473f834123860726712dca6259bb4e057a001)
- [2020-geilo-gpu-python](https://github.com/inducer/2020-geilo-gpu-python)

### Machine learning and Deep Learning

- https://towardsdatascience.com/fast-data-augmentation-in-pytorch-using-nvidia-dali-68f5432e1f5f
- https://ep2019.europython.eu/media/conference/slides/fX8dJsD-distributed-multi-gpu-computing-with-dask-cupy-and-rapids.pdf
- https://github.com/NVIDIA/DeepLearningExamples
- https://github.com/chagaz/hpc-ai-ml-2019
- [tensorflow tutorial](https://github.com/eth-cscs/SummerSchool2019/tree/master/topics/tensorflow)
- [AI cheatsheet](doc/ai_cheatsheet.pdf)
- [m2dsupsdlclass](https://github.com/m2dsupsdlclass/lectures-labs)

### Graphics / GPU

- https://raytracing.github.io/
- https://github.com/RayTracing/raytracing.github.io
- https://github.com/rogerallen/raytracinginoneweekendincuda : code très clean, super

### OpenMP

- https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
- https://github.com/OpenMP/Examples/tree/v4.5.0/sources
- https://ukopenmpusers.co.uk/wp-content/uploads/uk-openmp-users-2018-OpenMP45Tutorial_new.pdf
- https://www.nas.nasa.gov/hecc/assets/pdf/training/OpenMP4.5_3-20-19.pdf
- http://www.admin-magazine.com/HPC/Articles/OpenMP-Coding-Habits-and-GPUs?utm_source=AMEP

### OpenMP target

- How to build yourself clang with OpenMP target support for Nvidia GPUs
  - https://hpc-wiki.info/hpc/Building_LLVM/Clang_with_OpenMP_Offloading_to_NVIDIA_GPUs
  - //devmesh.intel.com/blog/724749/how-to-build-and-run-your-modern-parallel-code-in-c-17-and-openmp-4-5-library-on-nvidia-gpus
- https://www.openmp.org/wp-content/uploads/SC17-OpenMPBooth_jlarkin.pdf
- [OpenMP 5.0 for accelerators at GTC 2019](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9353-openmp-5-for-accelerators-and-what-comes-next.pdf)
- [LLVM/Clang based compiler for both AMD/NVidia GPUs](https://github.com/ROCm-Developer-Tools/aomp)
- [OpenMP target examples](https://github.com/pkestene/OMP-Offloading)

How to build clang++ with openmp target (off-loading) support ?

- https://devmesh.intel.com/blog/724749/how-to-build-and-run-your-modern-parallel-code-in-c-17-and-openmp-4-5-library-on-nvidia-gpus
- https://hpc-wiki.info/hpc/Building_LLVM/Clang_with_OpenMP_Offloading_to_NVIDIA_GPUs


### OpenACC

- [OpenACC Programming and Best Practices Guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)
- [PGI compiler - OpenACC getting started guide](https://www.pgroup.com/resources/docs/19.10/x86/openacc-gs/index.htm)
- https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/openacc/2-openacc-introduction.pdf?__blob=publicationFile
- [Introduction to GPU programming using OpenACC](https://www.fzj.de/ias/jsc/EN/Expertise/Services/Documentation/presentations/presentation-openacc_table.html)
- https://github.com/eth-cscs/SummerSchool2019/tree/master/topics/openacc
- https://developer.nvidia.com/openacc-overview-course
- https://perso.centrale-marseille.fr/~gchiavassa/visible/HPC/01%20-%20GR%20%20Intro%20to%20GPU%20programming%20V2%20OpenACC%20.pdf
- [Jeff Larkin (Nvidia) Introduction to OpenACC](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%201.pdf)
- [Jeff Larkin (Nvidia) OpenACC data management](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%202.pdf)
- [Jeff Larkin (Nvidia) OpenACC optimizations](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%203.pdf)
- https://www.pgroup.com/resources/docs/19.10/pdf/pgi19proftut.pdf
- https://github.com/OpenACCUserGroup/openacc_concept_strategies_book
- https://developer.nvidia.com/blog/solar-storm-modeling-gpu-openacc/

Which compiler with OpenAcc support ?
- [Nvidia/PGI compiler](https://developer.nvidia.com/hpc-sdk) is the oldest and probably more mature OpenACC compiler. 
- [GNU/gcc](https://www.openacc.org/tools/gcc-for-openacc) provided by [Spack](https://spack.readthedocs.io/en/latest/) is the easiest way to get started for OpenMP/OpenACC offload with the GNU compiler.

### C++17 and parallel STL for CPU/GPU

- [accelerating-standard-c-with-gpus-using-stdpar/](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/) for Nivia GPUs
- a real life example in CFD: [LULESH](https://github.com/LLNL/LULESH/tree/2.0.2-dev/stdpar)
- another reference in CFD [stdpar for Lattice Boltzmann simulation](https://arxiv.org/pdf/2010.11751.pdf) and its [companion code](https://gitlab.com/unigehpfs/stlbm)
- https://github.com/shwina/stdpar-cython/
- https://software.intel.com/content/www/us/en/develop/articles/get-started-with-parallel-stl.html

Which compiler ?
- [Nvidia/PGI compiler](https://developer.nvidia.com/hpc-sdk) for Nvidia GPUs
- GNU g++ version >= 9.1  (+ TBB) for multicore CPUs
- clang >= 10.0.1 for multicore CPUs
- [Intel OneApi HPC Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit.html)

### SYCL

- [Khronos](https://www.khronos.org/sycl/resources)
- [syclacademy](https://github.com/codeplaysoftware/syclacademy)
- [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples)
- [more oneAPI / SYCL samples](https://github.com/zjin-lcf/oneAPI-DirectProgramming)
- Compilers / toolchain
  * [codeplay](https://developer.codeplay.com/home/)
  * [Intel OneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html). If you want Nvidia GPU support, you'll have to rebuild llvm/clang from the [source code](https://github.com/intel/llvm), see [instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda); OneAPI DPC++ actually is a SYCL implementation + [extensions](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions) (Unified Shared Memory, Explicit SIMD, ...)
  * [triSYCL](https://github.com/triSYCL/triSYCL) for [Xilinx FPGA target](https://raw.githubusercontent.com/keryell/ronan/gh-pages/Talks/2019/2019-11-17-SC19-H2RC-keynote-SYCL/2019-11-17-SC19-H2RC-keynote-SYCL.pdf)

### Books on GPU programming / recommended reading

- [The CUDA Handbook: A Comprehensive Guide to GPU Programming](http://www.cudahandbook.com/), by Nicholas Wilt, Pearson Education.
- [CUDA by example](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685/ref=pd_bbs_sr_1/103-9839083-1501412?ie=UTF8&s=books&qid=1186428068&sr=1-1), by Sanders and Kandrot, Addison-Wesley, 2010. Also available in [pdf](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Learn CUDA programming](https://www.packtpub.com/eu/application-development/cuda-cookbook) by B. Sharma and J. Han, Packt Publishing, 2019
- Python + CUDA : https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA
- https://www.oreilly.com/library/view/hands-on-gpu-programming/9781788993913/ by Brian Tuomanen

### C++ resources

- [Discovering Modern C++: An Intensive Course for Scientists, Engineers, and Programmers](https://www.amazon.com/Discovering-Modern-Scientists-Programmers-Depth/dp/0134383583), and [companion github website](https://github.com/petergottschling/discovering_modern_cpp)
- https://github.com/changkun/modern-cpp-tutorial
- https://github.com/eth-cscs/examples_cpp
- https://github.com/mandliya/algorithms_and_data_structures
- https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cplusplus/cplusplus.pdf?__blob=publicationFile
- https://gitlab.maisondelasimulation.fr/tpadiole/hpcpp
- http://www.cppstdlib.com/
- http://101.lv/learn/C++/
- https://github.com/caveofprogramming/advanced-cplusplus
- https://en.cppreference.com/w/
- list of Lists of C++ related resources: https://github.com/fffaraz/awesome-cpp
- list of books on C++ : https://github.com/fffaraz/awesome-cpp/blob/master/books.md
- [C++ idioms](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms)
- [Design Patterns](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns) and [Book on design patterns for modern c++](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP)
- [Julich training on C++](https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cplusplus/cplusplus.pdf?)
- [CSCS computing center training on C++ videos](https://www.cscs.ch/publications/tutorials/2019/videos-of-workshop-advanced-c/)
- [CppCon](https://github.com/CppCon/) and [videos on YouTube](https://www.youtube.com/user/CppCon)
- [Bo Qiang YouTube channel on C++11](https://www.youtube.com/channel/UCEOGtxYTB6vo6MQ-WQ9W_nQ)
- https://github.com/TheAlgorithms/C-Plus-Plus
- [cours de C++ de l'université de Strasbourg](http://irma.math.unistra.fr/~franck/cours/Cpp1819/cpp1819English.html)


### high-level C++ libraries for programming GPUs

Alternate programming models for programming modern computing architectures in a performance portable way:

- introduction to [performance portability](https://performanceportability.org/perfport/overview/)
- https://github.com/arrayfire/arrayfire
- https://docs.nvidia.com/cuda/thrust/index.html
- https://github.com/kokkos/kokkos
- https://github.com/LLNL/RAJA et https://github.com/LLNL/RAJA-tutorials
- https://github.com/triSYCL/triSYCL
- https://github.com/codeplaysoftware/computecpp-sdk

### CMake

- [cmake-cookbook](https://github.com/dev-cafe/cmake-cookbook) and the [book](https://www.packtpub.com/application-development/cmake-cookbook)
- [Modern CMake tutorial](https://cliutils.gitlab.io/modern-cmake/)
- [template CMake project for CUDA](https://github.com/pkestene/cuda-proj-tmpl)
- [GPUs for science day](https://www.nersc.gov/assets/GPUs-for-Science-Day/jonathan-madsen.pdf)

### Git

- [Git cheatsheet](https://github.github.com/training-kit/)

### Misc

- [Udacity CS344 video archive](https://www.youtube.com/playlist?list=PLvvwOd40Y2t9lCTtCOQLJd9vLA2muyJuA)
- cuda related : https://gist.github.com/allanmac/f91b67c112bcba98649d - cuda_assert
- [FPGA, loop transformation, matrix multiplication](https://arxiv.org/pdf/1805.08288.pdf) 
- [Cycle du hype](https://fr.wikipedia.org/wiki/Cycle_du_hype)
- https://press3.mcs.anl.gov/atpesc/files/2019/08/ATPESC_2019_Dinner_Talk_8_8-7_Foster-Coding_the_Continuum.pdf

### Shell and command line skills

- Learn/improve your skill on Linux’s command line/Bash  
  e.g. http://swcarpentry.github.io/shell-novice/
- http://www.tldp.org/LDP/abs/html/
- http://www.epons.org/commandes-base-linux.php
- [The art of command line](https://github.com/jlevy/the-art-of-command-line)


### Blogs or newsletters on HPC

- https://www.nextplatform.com/
- subscribe blog/news letters on HPC; e.g. [Admin-magazine / HPC](http://www.admin-magazine.com/HPC/Articles)
- (En anglais) [Intel Parallel Universe Magazine](https://software.intel.com/en-us/parallel-universe-magazine)


# MOOC

- [Amazon](https://www.amazon.com/s?i=digital-text&rh=p_27%3ACuda+Education&s=relevancerank&text=Cuda+Education&ref=dp_byline_sr_ebooks_1)
- [udemy](https://www.udemy.com/)

# Projet

- Portage d'un code C++ de simulation des équations de Navier-Stokes par la méthode de Boltzmann sur réseau.
