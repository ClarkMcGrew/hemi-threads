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
#include "host_threads.h"

#ifdef HEMI_CUDA_COMPILER
#include "configure.h"
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
    ExecutionPolicy p;
    launch(p, f, args...);
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
    if (p.getGridSize() > 0 && p.getBlockSize() > 0 && !hemi::threads::gPool) {
        Kernel<<<p.getGridSize(),
             p.getBlockSize(),
             p.getSharedMemBytes(),
             p.getStream()>>>(f, args...);
        return;
    }
#ifndef HEMI_ALLOW_MISSING_DEVICE
    else {
        HEMI_LAUNCH_OUTPUT("hemi::launch: CUDA without available GPU");
        throw std::runtime_error("hemi::launch: GPU not available");
    }
#endif
#else
#ifndef HEMI_DISABLE_THREADS
    if (!hemi::threads::gPool and hemi::hostThreads != 1) {
        hemi::threads::gPool
            = std::make_unique<hemi::threads::ThreadPool>(hemi::hostThreads);
        hemi::hostThreads = hemi::threads::gPool->workerThreads();
    }
#endif
    if (!hemi::threads::gPool) {
       HEMI_LAUNCH_OUTPUT("Host launch (no GPU used) --"
                           << " direct call");
       f(args...);
       return;
    }
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used) --"
                       << " thread pool: " << hemi::hostThreads);
    hemi::threads::gPool->kernel(f, args...);
#endif
}

//
// Automatic launch functions for __global__ kernel function pointers: CUDA only
//

template <typename... Arguments>
void cudaLaunch(void(*f)(Arguments... args), Arguments... args)
{
    ExecutionPolicy p;
    cudaLaunch(p, f, args...);
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
    if (p.getGridSize() > 0 && p.getBlockSize() > 0 && !hemi::threads::gPool) {
        f<<<p.getGridSize(),
            p.getBlockSize(),
            p.getSharedMemBytes(),
            p.getStream()>>>(args...);
        return;
    }
#ifndef HEMI_ALLOW_MISSING_DEVICE
    else {
        HEMI_LAUNCH_OUTPUT("hemi::cudaLaunch: CUDA without available GPU");
        throw std::runtime_error("cudaLaunch: GPU not available");
    }
#endif
#else
#ifndef HEMI_DISABLE_THREADS
    if (!hemi::threads::gPool and hemi::hostThreads != 1) {
        hemi::threads::gPool
            = std::make_unique<hemi::threads::ThreadPool>(hemi::hostThreads);
        hemi::hostThreads = hemi::threads::gPool->workerThreads();
    }
#endif
    if (!hemi::threads::gPool) {
       HEMI_LAUNCH_OUTPUT("Host cudaLaunch (no GPU used) --"
                           << " direct call");
       f(args...);
       return;
    }
    HEMI_LAUNCH_OUTPUT("Host cudaLaunch (no GPU used) --"
                       << " thread pool: " << hemi::hostThreads);
    hemi::threads::gPool->kernel(f, args...);
#endif
}

inline void setHostThreads(int i) {
   hemi::hostThreads = i;
}
} // namespace hemi
