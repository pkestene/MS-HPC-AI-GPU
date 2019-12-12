1. Install pgi compiler; have a look at [PGI getting started guide](https://www.pgroup.com/resources/docs/19.10/pdf/openacc19_gs.pdf)
2. copy samples
   ```bash
   cp -a /opt/pgi/linux86-64/2019/examples/OpenACC/SDK/src/* .
   ```
3. run pgaccelinfo to get info about PGI Default Target

4. The minimal PGI flags for NVIDIA GPU: 
  * -acc -Minfo=accel -ta=tesla:cc60
  * -ta=tesla:lineinfo
  * -Minfo=all (more feedback from compiler)
  * -ta=tesla:managed (unified memory)
  
5. The minimal PGI flags for multicore CPU:
  * -acc -ta=multicore

# Parallel Construct

Defines the region of the program that should be compiled for parallel execution on the accelerator device.

# Kernels Construct

Defines the region of the program that should be compiled into a *sequence* of kernels for execution on the accelerator device.

## C/C++

```c++
#pragma acc kernels [clause [[,] clause]...] new-line
{ structured block }
```

## Fortran

```fortran
!$acc kernels [clause [[,] clause]...]
structured block
!$acc end kernels
```

Any data clause is allowed.

## other clauses

```c++
if( condition )
```

When the condition is nonzero or .TRUE. the kernels region will execute on the accelerator; otherwise, it will execute on the host.

```c++
async( expression )
```

The kernels region executes asynchronously with the host.


# Data Construct

An accelerator *data* construct defines a region of the program within which data is accessible by the accelerator.

```c++
// C/C++
#pragma acc data [clause[[,] clause]...] new-line
{ structured block }
```

```fortran
! Fortran
!$acc data [clause[[,] clause]...]
structured block
!$acc end data```

# Data Clauses

The description applies to the clauses used on parallel constructs, kernels constructs, data constructs, declare constructs, and update directives.

## copy( list )

Allocates the data in list on the accelerator and copies the data from the host to the accelerator when entering the region, and copies the data from the accelerator to the host when exiting the region.

## copyin( list )

Allocates the data in list on the accelerator and copies the data from the host to the accelerator when entering the region.

## copyout( list )

Allocates the data in list on the accelerator and copies the data from the accelerator to the host when exiting the region.

## create( list )

Allocates the data in list on the accelerator, but does not copy data between the host and device.

## present( list )

The data in list must be already present on the accelerator, from some containing data region; that accelerator copy is found and used.


# Documentation

- https://www.pgroup.com/resources/docs/19.10/pdf/pgi19proftut.pdf
