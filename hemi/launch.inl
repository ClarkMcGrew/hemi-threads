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
void launch([[maybe_unused]] const ExecutionPolicy &policy, Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, Kernel<Function, Arguments...>));
    HEMI_LAUNCH_OUTPUT("hemi::launch with grid size of " << p.getGridSize()
                       << " and block size of " << p.getBlockSize());
    do {
#ifdef HEMI_CUDA_CHECK_ERRORS
        if (p.getGridSize() < 1) {
            std::cerr << "hemi::launch: Grid size is zero"
                      << " (GPU may not be available)" << std::endl;
            throw std::runtime_error("hemi::launch: Invalid grid size");
        }
        if (p.getBlockSize() < 1) {
            std::cerr << "hemi::launch: Grid size is zero"
                      << " (GPU may not be available)" << std::endl;
            throw std::runtime_error("hemi::launch: Invalid block size");
        }
#endif
        Kernel<<<p.getGridSize(),
             p.getBlockSize(),
             p.getSharedMemBytes(),
             p.getStream()>>>(f, args...);
#ifdef HEMI_CUDA_CHECK_ERRORS
        if (cudaSuccess != checkCudaErrors()) {
            std::cerr << "hemi::launch: Kernel launch error" << std::endl;
            throw std::runtime_error("hemi::launch: Kernel launch error");
        }
#endif
        return;
    } while (false);
#else
#ifndef HEMI_DISABLE_THREADS
    if (!hemi::threads::gPool and hemi::threads::number != 1) {
        hemi::threads::gPool.reset(new hemi::threads::ThreadPool(
                                       hemi::threads::number));
        hemi::threads::number = hemi::threads::gPool->workerThreads();
    }
#endif
    if (!hemi::threads::gPool) {
       HEMI_LAUNCH_OUTPUT("Host launch (no GPU used) --"
                           << " direct call");
       f(args...);
       return;
    }
    HEMI_LAUNCH_OUTPUT("Host launch (no GPU used) --"
                       << " thread pool: " << hemi::threads::number);
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
void cudaLaunch([[maybe_unused]] const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, f));
    HEMI_LAUNCH_OUTPUT("cudaLaunch: with grid size of " << p.getGridSize()
                       << " and block size of " << p.getBlockSize());
    do {
#ifdef HEMI_CUDA_CHECK_ERRORS
        if (p.getGridSize() < 1) {
            std::cerr << "hemi::cudaLaunch: Grid size is zero"
                      << " (GPU may not be available)" << std::endl;
            throw std::runtime_error("hemi::cudaLaunch: Invalid grid size");
        }
        if (p.getBlockSize() < 1) {
            std::cerr << "hemi::cudaLaunch: Grid size is zero"
                      << " (GPU may not be available)" << std::endl;
            throw std::runtime_error("hemi::cudaLaunch: Invalid block size");
        }
#endif
        f<<<p.getGridSize(),
            p.getBlockSize(),
            p.getSharedMemBytes(),
            p.getStream()>>>(args...);
#ifdef HEMI_CUDA_CHECK_ERRORS
        if (cudaSuccess != checkCudaErrors()) {
            std::cerr << "hemi::cudaLaunch: Kernel launch error" << std::endl;
            throw std::runtime_error("hemi::cudaLaunch: Kernel launch error");
        }
#endif
       return;
    } while (false);
#else
#ifndef HEMI_DISABLE_THREADS
    if (!hemi::threads::gPool and hemi::threads::number != 1) {
        hemi::threads::gPool.reset(new hemi::threads::ThreadPool(
                                       hemi::threads::number));
        hemi::threads::number = hemi::threads::gPool->workerThreads();
    }
#endif
    if (!hemi::threads::gPool) {
       HEMI_LAUNCH_OUTPUT("Host cudaLaunch (no GPU used) --"
                           << " direct call");
       f(args...);
       return;
    }
    HEMI_LAUNCH_OUTPUT("Host cudaLaunch (no GPU used) --"
                       << " thread pool: " << hemi::threads::number);
    hemi::threads::gPool->kernel(f, args...);
#endif
}

} // namespace hemi
