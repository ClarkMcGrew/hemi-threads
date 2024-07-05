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

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
#include "range/range.hpp"
#include "hemi.h"
#include "device_api.h"

namespace hemi {

    // type alias to simplify typing...
    template<typename T>
    using step_range = typename util::lang::range_proxy<T>::step_range_proxy;

#if not defined(HEMI_CUDA_DISABLE) || defined(HEMI_HOST_WITH_STRIDE)
    template <typename T>
    HEMI_DEV_CALLABLE_INLINE
    step_range<T> grid_stride_range(T begin, T end) {
        begin += hemi::globalThreadIndex();
        return util::lang::range(begin, end).step(hemi::globalThreadCount());
    }
#else
    template <typename T>
    HEMI_DEV_CALLABLE_INLINE
    step_range<T> grid_stride_range(T begin, T end) {
        if (end < begin) std::runtime_error("Invalid grid_stride_range");
        T steps = 1 + (end-begin)/hemi::globalThreadCount();
        T first = begin + hemi::globalThreadIndex()*steps;
        T last = first + steps;
        if (last > end) last = end;
        return util::lang::range(first, last).step(1);
    }
#endif

}
/////////////////////////////////////////////////////////////////
