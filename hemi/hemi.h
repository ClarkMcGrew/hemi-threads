///////////////////////////////////////////////////////////////////////////////
//
// "Hemi" CUDA Portable C/C++ Utilities
//
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md)
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#pragma once

/* HEMI_VERSION encodes the version number of the HEMI utilities.
 *
 *   HEMI_VERSION / 100000 is the major version.
 *   HEMI_VERSION / 100 % 1000 is the minor version.
 */
#define HEMI_VERSION 200000

// Note: when compiling on a system without CUDA installed
// be sure to define this macro. Can also be used to disable CUDA
// device execution on systems with CUDA installed.
// #define HEMI_CUDA_DISABLE

#ifdef HEMI_CUDA_DISABLE
#warning HEMI: HEMI_CUDA_DISABLE is defined so CUDA usage is disabled
#endif

// The flag that is defined in the file where global variables should be
// defined.  This is only needed with the feature flag __cpp_inline_variables
// (i.e. pre-C++17) is not defined.  It can only occur in *ONE* C++ source
// file (this is required to meet the One Definition Rule).
#ifdef HEMI_COMPILE_DEFINITIONS
#ifdef __cpp_inline_variables
#undef HEMI_COMPILE_DEFINITIONS   // Not required so undefine
#else
#warning HEMI_COMPILE_DEFINITIONS is set.  Globals will be defined.
#endif
#endif

// Helper macro for defining functors that can be launched as kernels
#define HEMI_KERNEL_FUNCTION(name, ...)                              \
  struct name {                                                      \
     HEMI_DEV_CALLABLE_MEMBER void operator()(__VA_ARGS__) const;    \
  };                                                                 \
  HEMI_DEV_CALLABLE_MEMBER void name::operator()(__VA_ARGS__) const

#if !defined(HEMI_CUDA_DISABLE) && defined(__CUDACC__) // CUDA compiler

  #define HEMI_CUDA_COMPILER              // to detect CUDACC compilation
  #define HEMI_LOC_STRING "Device"

  #ifdef __CUDA_ARCH__
    #define HEMI_DEV_CODE                 // to detect device compilation
  #endif

  #define HEMI_KERNEL(name)               __global__ void name ## _kernel
  #define HEMI_KERNEL_NAME(name)          name ## _kernel

  #if defined(DEBUG) || defined(_DEBUG) || defined(HEMI_DEBUG)
    #warning HEMI debug compilation uses synchronous device operations
    #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
    do {                                                                     \
        name ## _kernel<<< (gridDim), (blockDim), (sharedBytes), (streamId) >>>\
            (__VA_ARGS__);                                                     \
        checkCudaErrors();                                                     \
    } while(0)
  #else
    #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
        name ## _kernel<<< (gridDim) , (blockDim), (sharedBytes), (streamId) >>>(__VA_ARGS__)
  #endif

  #define HEMI_LAUNCHABLE                 __global__
  #define HEMI_LAMBDA                     __device__
  #define HEMI_DEV_CALLABLE               __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE        __host__ __device__ inline
  #define HEMI_DEV_CALLABLE_MEMBER        __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER __host__ __device__ inline

  // Memory specifiers
  #define HEMI_MEM_DEVICE                 __device__

  // Stream type
  typedef cudaStream_t hemiStream_t;

  // Constants: declares both a device and a host copy of this constant
  // static and extern flavors can be used to declare static and extern
  // linkage as required.
  #define HEMI_DEFINE_CONSTANT(def, value) \
      __constant__ def ## _devconst = value; \
      def ## _hostconst = value
  #define HEMI_DEFINE_STATIC_CONSTANT(def, value) \
      static __constant__ def ## _devconst = value; \
      static def ## _hostconst = value
  #define HEMI_DEFINE_EXTERN_CONSTANT(def) \
      extern __constant__ def ## _devconst; \
      extern def ## _hostconst

  // use to access device constant explicitly
  #define HEMI_DEV_CONSTANT(name) name ## _devconst

  // use to access a constant defined with HEMI_DEFINE_*_CONSTANT
  // automatically chooses either host or device depending on compilation
  #ifdef HEMI_DEV_CODE
    #define HEMI_CONSTANT(name) name ## _devconst
  #else
    #define HEMI_CONSTANT(name) name ## _hostconst
  #endif

  #if !defined(HEMI_ALIGN)
    #define HEMI_ALIGN(n) __align__(n)
  #endif

#else             // host compiler
  #define HEMI_HOST_COMPILER              // to detect non-CUDACC compilation
  #define HEMI_LOC_STRING "Host"

  #define HEMI_KERNEL(name)               void name
  #define HEMI_KERNEL_NAME(name)          name
  #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) name(__VA_ARGS__)

  #define HEMI_LAUNCHABLE
  #define HEMI_LAMBDA
  #define HEMI_DEV_CALLABLE
  #define HEMI_DEV_CALLABLE_INLINE        inline
  #define HEMI_DEV_CALLABLE_MEMBER
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER inline

  // memory specifiers
  #define HEMI_MEM_DEVICE

  // Stream type
  typedef int hemiStream_t;

  #define HEMI_DEFINE_CONSTANT(def, value) def ## _hostconst = value
  #define HEMI_DEFINE_STATIC_CONSTANT(def, value) static def ## _hostconst = value
  #define HEMI_DEFINE_EXTERN_CONSTANT(def) extern def ## _hostconst

  #undef HEMI_DEV_CONSTANT // requires NVCC, so undefined here!
  #define HEMI_CONSTANT(name) name ## _hostconst

  #if !defined(HEMI_ALIGN)

    #if defined(__GNUC__)
      #define HEMI_ALIGN(n) __attribute__((aligned(n)))
    #elif defined(_MSC_VER)
      #define HEMI_ALIGN(n) __declspec(align(n))
    #else
      #error "Please provide a definition of HEMI_ALIGN for your host compiler!"
    #endif

  #endif

#endif

#if defined(__cpp_inline_variables)
// Global inline variables are supported, so use them.  This lets the linker
// choose which instance of the variable will be created in the executable.
// See host_threads.h for a usage example.
#define HEMI_INLINE_VARIABLE(declaration,init) inline declaration init
#elif defined(HEMI_COMPILE_DEFINITIONS)
// This is the "definition" complation unit, so global variables are defined
// and initialized.
#define HEMI_INLINE_VARIABLE(declaration,init) declaration init
#else
// Inline variables are not supported.  That means that the global variables
// cannot be initialized here, and that the will have to be defined once in
// the executable (usually the main program source file).  This is done by
// defining HEMI_COMPILE_DEFINITIONS in *ONE* c++ source file that will be
// lined into the executable (or added to the library).  This is mostly
// relevant for C++11 and C++14 since C++17 and later support inline
// variables.
#define HEMI_INLINE_VARIABLE(declaration,init) extern declaration
#endif

#include "hemi_error.h"
#include "host_threads.h"

namespace hemi {

    inline hemi::Error_t deviceSynchronize()
    {
        if (hemi::threads::gPool) hemi::threads::gPool->wait();
#ifndef HEMI_CUDA_DISABLE
        else if (cudaSuccess != checkCuda(cudaDeviceSynchronize())) {
             return hemi::cudaError;
        }
#endif
        return hemi::success;
    }

} // namespace hemi

#ifdef HEMI_CUDA_DISABLE
namespace hemi {
    inline bool deviceAvailable(bool dump = false) {
         if (dump) {
              std::cout << "HEMI DEVICE COUNT: No Device" << std::endl;
         }
         return false;
    }
}
#else
#include <cuda_runtime_api.h>
namespace hemi {
    inline bool deviceAvailable(bool dump = false) {
         cudaError_t status;
         int devCount;
         status = cudaGetDeviceCount(&devCount);
         if (status != cudaSuccess) {
              if (dump) std::cout << "HEMI DEVICE COUNT: No Device" << std::endl;
              return false;
         }

         int devId;
         status = cudaGetDevice(&devId);
         if (status != cudaSuccess) {
              if (dump) std::cout << "HEMI DEVICE COUNT: No Device" << std::endl;
              return false;
         }

         cudaDeviceProp prop;
         status = cudaGetDeviceProperties(&prop, devId);
         if (status != cudaSuccess) {
              if (dump) std::cout << "HEMI DEVICE COUNT: No Device" << std::endl;
              return false;
         }

         if (dump) {
              std::cout << "HEMI DEVICE COUNT:         " << devCount << std::endl;
              std::cout << "HEMI DEVICE ID:            " << devId << std::endl;
              std::cout << "HEMI DEVICE NAME:          " << prop.name << std::endl;
              std::cout << "HEMI COMPUTE CAPABILITY:   " << prop.major << "." << prop.minor << std::endl;
              std::cout << "HEMI PROCESSORS:           " << prop.multiProcessorCount << std::endl;
              std::cout << "HEMI PROCESSOR THREADS:    " << prop.maxThreadsPerMultiProcessor << std::endl;
              std::cout << "HEMI MAX THREADS:          " << prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount << std::endl;
              std::cout << "HEMI THREADS PER BLOCK:    " << prop.maxThreadsPerBlock << std::endl;
              std::cout << "HEMI BLOCK MAX DIM:        " << "X:" << prop.maxThreadsDim[0]
                        << " Y:" << prop.maxThreadsDim[1]
                        << " Z:" << prop.maxThreadsDim[2] << std::endl;
              std::cout << "HEMI GRID MAX DIM:         " << "X:" << prop.maxGridSize[0]
                        << " Y:" << prop.maxGridSize[1]
                        << " Z:" << prop.maxGridSize[2] << std::endl;
              std::cout << "HEMI WARP:                 " << prop.warpSize << std::endl;
              std::cout << "HEMI CLOCK:                " << prop.clockRate << std::endl;
              std::cout << "HEMI GLOBAL MEM:           " << prop.totalGlobalMem << std::endl;
              std::cout << "HEMI SHARED MEM:           " << prop.sharedMemPerBlock << std::endl;
              std::cout << "HEMI L2 CACHE MEM:         " << prop.l2CacheSize << std::endl;
              std::cout << "HEMI CONST MEM:            " << prop.totalConstMem << std::endl;
              std::cout << "HEMI MEM PITCH:            " << prop.memPitch << std::endl;
              std::cout << "HEMI REGISTERS:            " << prop.regsPerBlock << std::endl;
         }

         if (prop.totalGlobalMem < 1) return false;
         if (prop.multiProcessorCount < 1) return false;
         if (prop.maxThreadsPerBlock < 1) return false;
         return true;
    }
} // namespace hemi
#endif
