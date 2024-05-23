#include "hemi_test.h"
#include "test_CAS.h"
#include "hemi/array.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include <algorithm>

#if defined(HEMI_THREADS_DISABLE) && defined(HEMI_CUDA_DISABLE)
#define ArrayTest ArrayTestSingle
#elif defined(HEMI_CUDA_DISABLE)
#define ArrayTest ArrayTestHost
#else
#define ArrayTest ArrayTestDevice
#endif

#ifndef HEMI_DEV_CODE
namespace {
    inline void hostAtomicAdd(float* value, float increment) {
        float expected = *value;
        float updated;
        float trials = 0;
        do {
            if (++trials > 1000) std::runtime_error("Failed hostAtomicAdd");
            updated = expected + increment;
        } while (not test_CAS_float(value,&expected,updated));
    }
    inline void hostAtomicSet(float* value, float newValue) {
        float expected = *value;
        float updated;
        float trials = 0;
        do {
            if (++trials > 1000) std::runtime_error("Failed hostAtomicAdd");
            updated = newValue;
        } while (not test_CAS_float(value,&expected,updated));
    }
}
#endif

TEST(ArrayTest, CreatesAndFillsArrayOnHost)
{

    const int n = 50;
    const float val = 3.14159f;
    hemi::Array<float> data(n);

    ASSERT_EQ(data.size(), n);

    float *ptr = data.writeOnlyHostPtr();
    ASSERT_NE(ptr,nullptr);

    std::fill(ptr, ptr+n, val);

    for(int i = 0; i < n; i++) {
        ASSERT_EQ(val, data.readOnlyHostPtr()[i]);
    }

}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMIFill, float* ptr, int N, float val) {
        for (int i : hemi::grid_stride_range(0,N)) {
            ptr[i] = val;
        }
    }
}


TEST(ArrayTest, CreatesAndFillsArrayOnDevice)
{
    const int n = 50;
    const float val = 3.14159f;
    hemi::Array<float> data(n);

    ASSERT_EQ(data.size(), n);

    HEMIFill fillArray;
    hemi::launch(fillArray,data.writeOnlyPtr(),n,val);

    for(int i = 0; i < n; i++) {
        EXPECT_EQ(val, data.readOnlyPtr(hemi::host)[i])
            << "Array value mismatched";
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISquare, float* ptr, int N) {
        for (int i : hemi::grid_stride_range(0,N)) {
            ptr[i] = ptr[i]*ptr[i];
        }
    }
}

TEST(ArrayTest, FillsOnHostModifiesOnDevice)
{
    const int n = 100;
    float val = 2.0;
    hemi::Array<float> data(n);

    ASSERT_EQ(data.size(), n);

    float *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+n, val);

    HEMISquare squareArray;
    hemi::launch(squareArray,data.ptr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;

    for(int i = 0; i < n; i++) {
        float result = data.readOnlyPtr(hemi::host)[i];
        EXPECT_EQ(val,result)
            << "Mismatch at element " << i
            << " current: " << data.readOnlyPtr(hemi::host)[i];
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMICoverage, float* ptr, int N) {
        for (int i : hemi::grid_stride_range(0,N)) {
#ifndef HEMI_DEV_CODE
            hostAtomicAdd(&ptr[i],1.0*i+1.0);
#else
            atomicAdd(&ptr[i], 1.0*i+1.0);
#endif
        }
    }
}

TEST(ArrayTest, CheckCoverageInKernel)
{
    const int n = 100;
    hemi::Array<float> data(n);

    ASSERT_EQ(data.size(), n);

    float *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+n, 0.0);

    HEMICoverage kernelCoverage;
    hemi::launch(kernelCoverage,data.ptr(),data.size());
    hemi::launch(kernelCoverage,data.writeOnlyPtr(),data.size());
    hemi::launch(kernelCoverage,data.writeOnlyPtr(),data.size());

    for(int i = 0; i < n; i++) {
        EXPECT_EQ(3.0*(i+1), data.readOnlyPtr(hemi::host)[i])
            << "Elements not covered by kernel";
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}
