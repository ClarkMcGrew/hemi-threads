#include "gtest/gtest.h"
#include "hemi/launch.h"
#include "hemi/device_api.h"
#include "hemi/grid_stride_range.h"
#include "unistd.h"

#if defined(HEMI_THREADS_DISABLE) && defined(HEMI_CUDA_DISABLE)
#define PortableLaunchTest PortableLaunchTestSingle
#elif defined(HEMI_CUDA_DISABLE)
#define PortableLaunchTest PortableLaunchTestHost
#else
#define PortableLaunchTest PortableLaunchTestDevice
#endif

HEMI_MEM_DEVICE int result;
HEMI_MEM_DEVICE int rGDim;
HEMI_MEM_DEVICE int rBDim;

template <typename T, typename... Arguments>
HEMI_DEV_CALLABLE
T first(T f, Arguments...) {
	return f;
}

template <typename... Arguments>
struct k {
	HEMI_DEV_CALLABLE_MEMBER void operator()(Arguments... args) const {
		result = first(args...); //sizeof...(args);
#ifdef HEMI_DEV_CODE
		rGDim = 1;//gridDim.x;
		rBDim = 1;//blockDim.x;
#endif
	}
};

struct slowKernel {
    HEMI_DEV_CALLABLE_MEMBER void operator()(int increment) const {
        // printf("start %d\n", hemi::globalThreadIndex());
        double sum = 0.0;
        for (int i : hemi::grid_stride_range(0,1000)) {
            for (int j = 0; j < 10000; ++j)  sum = sum + increment;
            // if (sum >10) printf("slow %d %d %f\n", hemi::globalThreadIndex(), i, sum);
        }
    }
};

TEST(PortableLaunchTest, KernelFunction_AutoConfig) {
    k<int> kernel;
    hemi::launch(kernel, 1);
}

TEST(PortableLaunchTest, KernelFunction_Slow) {
    slowKernel kernel;
    hemi::launch(kernel, 1);
}
