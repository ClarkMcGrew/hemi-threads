#include "gtest/gtest.h"

#ifndef HEMI_CUDA_DISABLE
#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));
#else
#define ASSERT_SUCCESS(res)
#define ASSERT_FAILURE(res)
#endif
