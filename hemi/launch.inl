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

#include "kernel.h"

#ifdef HEMI_CUDA_COMPILER
#include "configure.h"
#else
#include "host_threads.h"
#endif

namespace hemi {

// Number of threads to be used on a host.  Defaults the std::threads::hardware_concurrency

inline int hostThreads{0};

//
// Automatic Launch functions for closures (functor or lambda)
//
template <typename Function, typename... Arguments>
void launch(Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p;
    launch(p, f, args...);
#else
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used)");
    if (hemi::hostThreads == 1) {
       f(args...);
       return;
    }
    if (!hemi::threads::gPool) {
        hemi::threads::gPool
            = std::make_unique<hemi::threads::ThreadPool>(hemi::hostThreads);
    }
    hemi::threads::gPool->kernel(f, args...);
#endif
}

//
// Launch with explicit (or partial) configuration
//
template <typename Function, typename... Arguments>
void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, Kernel<Function, Arguments...>));
    HEMI_LAUNCH_OUTPUT("hemi::launch with grid size of " << p.getGridSize()
                       << " and block size of " << p.getBlockSize());
    if (p.getGridSize() > 0 && p.getBlockSize() > 0) {
        Kernel<<<p.getGridSize(),
             p.getBlockSize(),
             p.getSharedMemBytes(),
             p.getStream()>>>(f, args...);
    }
    else {
        HEMI_LAUNCH_OUTPUT("hemi::launch: CUDA without available GPU");
        throw std::runtime_error("hemi::launch: GPU not available");
    }
#else
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used)");
    launch(f, args...);
#endif
}

//
// Automatic launch functions for __global__ kernel function pointers: CUDA only
//

template <typename... Arguments>
void cudaLaunch(void(*f)(Arguments... args), Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p;
    cudaLaunch(p, f, args...);
#else
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used)");
    if (hemi::hostThreads == 1) {
       f(args...);
       return;
    }
    if (!hemi::threads::gPool) {
        hemi::threads::gPool
            = std::make_unique<hemi::threads::ThreadPool>(hemi::hostThreads);
    }
    hemi::threads::gPool->kernel(f, args...);
#endif
}

//
// Launch __global__ kernel function with explicit configuration
//
template <typename... Arguments>
void cudaLaunch(const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, f));
    HEMI_LAUNCH_OUTPUT("cudaLaunch: with grid size of " << p.getGridSize()
                       << " and block size of " << p.getBlockSize());
    if (p.getGridSize() > 0 && p.getBlockSize() > 0) {
        f<<<p.getGridSize(),
            p.getBlockSize(),
            p.getSharedMemBytes(),
            p.getStream()>>>(args...);
    }
    else {
        HEMI_LAUNCH_OUTPUT("hemi::cudaLaunch: CUDA without available GPU");
        throw std::runtime_error("cudaLaunch: GPU not available");
    }
#else
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used)");
    cudaLaunch(f, args...);
#endif
}

inline void setHostThreads(int i) {
   hemi::hostThreads = i;
}
} // namespace hemi
