///////////////////////////////////////////////////////////////////////////////
//
// "Hemi" CUDA Portable C/C++ Utilities
//
// Copyright 2012-2014 NVIDIA Corporation
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

#ifdef HEMI_LAUNCH_DEBUG
#warning hemi::launch -- Compiled with debugging
#include <iostream>
#define HEMI_LAUNCH_OUTPUT(arg) std::cout << arg << std::endl
#else
#define HEMI_LAUNCH_OUTPUT(arg) /* nothing */
#endif

#include "kernel.h"
#include "execution_policy.h"

namespace hemi {

    class ExecutionPolicy; // forward decl

    // Automatic parallel launch for function object
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args);

    // Launch function object with an explicit execution policy / configuration
    template <typename Function, typename... Arguments>
    void launch(const ExecutionPolicy &p, Function f, Arguments... args);

    // Automatic parallel launch for CUDA __global__ functions
    template <typename... Arguments>
    void cudaLaunch(void(*f)(Arguments...), Arguments... args);

    // Launch __global__ function with an explicit execution policy / configuration
    template <typename... Arguments>
    void cudaLaunch(const ExecutionPolicy &p, void(*f)(Arguments...), Arguments... args);

    // Number of host threads to be used (defaults to hardware_concurrency).
    // Set less than or equal to zero for default value.  This must be fixed
    // before first launch.
    void setHostThreads(int i);

}

#include "launch.inl"
