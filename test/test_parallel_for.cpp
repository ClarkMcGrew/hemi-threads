#include "test_CAS.h"
#include "hemi_test.h"
#include "hemi/parallel_for.h"
#include <cuda_runtime_api.h>

#if defined(HEMI_THREADS_DISABLE) && defined(HEMI_CUDA_DISABLE)
#define ParallelForTest ParallelForTestSingle
#elif defined(HEMI_CUDA_DISABLE)
#define ParallelForTest ParallelForTestHost
#else
#define ParallelForTest ParallelForTestDevice
#endif

#ifndef HEMI_DEV_CODE
namespace {
    inline void hostAtomicAdd(int* value, int increment) {
        int expected = *value;
        int updated;
        int trials = 0;
        do {
            if (++trials > 1000) std::runtime_error("Failed hostAtomicAdd");
            updated = expected + increment;
        } while (not test_CAS_int(value,&expected,updated));
    }
    inline void hostAtomicSet(int* value, int newValue) {
        int expected = *value;
        int updated;
        int trials = 0;
        do {
            if (++trials > 1000) std::runtime_error("Failed hostAtomicAdd");
            updated = newValue;
        } while (not test_CAS_int(value,&expected,updated));
    }
}
#endif

// Separate function because __device__ lambda can't be declared
// inside a private member function, and TEST() defines TestBody()
// private to the test class
void runParallelFor([[maybe_unused]] int instance, int loop, int *count, int *gdim, int *bdim)
{
#ifdef HEMI_PARALLEL_FOR_ENABLED
    hemi::parallel_for(0, loop, [=] HEMI_LAMBDA (int i) {
#ifndef HEMI_DEV_CODE
        hostAtomicSet(gdim,hemi::globalBlockCount());
        hostAtomicSet(bdim,hemi::localThreadCount());
        hostAtomicAdd(count,1);
#else
        *gdim = hemi::globalBlockCount();
        *bdim = hemi::localThreadCount();
        atomicAdd(count, 1);
#endif

    });
#else
#warning hemi::parallel_for not enabled causing test to fail.
    throw std::runtime_error("hemi::parallel_for not enabled");
#endif
}

void runParallelForEP([[maybe_unused]] int instance, const hemi::ExecutionPolicy &ep, int loop, int *count, int *gdim, int *bdim)
{
#ifdef HEMI_PARALLEL_FOR_ENABLED
    hemi::parallel_for(ep, 0, loop, [=] HEMI_LAMBDA (int) {

#ifndef HEMI_DEV_CODE
        hostAtomicSet(gdim,hemi::globalBlockCount());
        hostAtomicSet(bdim,hemi::localThreadCount());
        hostAtomicAdd(count,1);
#else
        *gdim = hemi::globalBlockCount();
        *bdim = hemi::localThreadCount();
        atomicAdd(count, 1);
#endif

    });
#else
#warning hemi::parallel_for not enabled causing test to fail.
    throw std::runtime_error("hemi::parallel_for not enabled");
#endif
}

class ParallelForTest : public ::testing::Test {
protected:
    virtual void SetUp() {
#ifdef HEMI_CUDA_COMPILER
        ASSERT_SUCCESS(cudaMalloc(&dCount, sizeof(int)));
        ASSERT_SUCCESS(cudaMalloc(&dBdim, sizeof(int)));
        ASSERT_SUCCESS(cudaMalloc(&dGdim, sizeof(int)));

        int devId;
        ASSERT_SUCCESS(cudaGetDevice(&devId));
        ASSERT_SUCCESS(cudaDeviceGetAttribute(&smCount,
                                              cudaDevAttrMultiProcessorCount,
                                              devId));
#else
        dCount = new int;
        dBdim = new int;
        dGdim = new int;

        smCount = 1;
#endif
    }

    virtual void TearDown() {
#ifdef HEMI_CUDA_COMPILER
        ASSERT_SUCCESS(cudaFree(dCount));
        ASSERT_SUCCESS(cudaFree(dBdim));
        ASSERT_SUCCESS(cudaFree(dGdim));
#else
        hemi::deviceSynchronize();
        delete dCount;
        delete dBdim;
        delete dGdim;
#endif
    }

    void Zero() {
#ifdef HEMI_CUDA_COMPILER
        ASSERT_SUCCESS(cudaMemset(dCount, 0, sizeof(int)));
        ASSERT_SUCCESS(cudaMemset(dBdim, 57, sizeof(int)));
        ASSERT_SUCCESS(cudaMemset(dGdim, 57, sizeof(int)));
#else
        hemi::deviceSynchronize();
        hostAtomicSet(dCount,0);
        hostAtomicSet(dBdim,57);
        hostAtomicSet(dGdim,57);
#endif
        count = 0;
        bdim = 0;
        gdim = 0;
    }

    void CopyBack() {
#ifdef HEMI_CUDA_COMPILER
        ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
        ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
        ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));
#else
        hemi::deviceSynchronize();
        count = *dCount;
        bdim = *dBdim;
        gdim = *dGdim;
#endif
    }

    int smCount;

    int *dCount;
    int *dBdim;
    int *dGdim;

    int count;
    int bdim;
    int gdim;
};

TEST_F(ParallelForTest, ComputesCorrectSum) {
    Zero();
    int loops = 1000;
    runParallelFor(1, loops, dCount, dGdim, dBdim);
    CopyBack();
    ASSERT_EQ(count, loops);
}


TEST_F(ParallelForTest, AutoConfigMaximalLaunch) {
    Zero();
    runParallelFor(2, 1000, dCount, dGdim, dBdim);
    CopyBack();

    ASSERT_GE(gdim, smCount);
    ASSERT_EQ(gdim%smCount, 0);
#ifdef HEMI_CUDA_COMPILER
    ASSERT_GE(bdim, 32);
#else
    ASSERT_EQ(bdim, 1);
#endif
}

TEST_F(ParallelForTest, ExplicitBlockSize)
{
    Zero();
    hemi::ExecutionPolicy ep;
    ep.setBlockSize(128);
    runParallelForEP(1, ep, 1000, dCount, dGdim, dBdim);
    CopyBack();

    ASSERT_GE(gdim, smCount);
    ASSERT_EQ(gdim%smCount, 0);
#ifdef HEMI_CUDA_COMPILER
    ASSERT_EQ(bdim, 128);
#else
    ASSERT_EQ(bdim, 1);
#endif
}

TEST_F(ParallelForTest, ExplicitGridSize)
{
    Zero();
    hemi::ExecutionPolicy ep;
    ep.setGridSize(100);
    runParallelForEP(2, ep, 1000, dCount, dGdim, dBdim);
    CopyBack();

#ifdef HEMI_CUDA_COMPILER
    ASSERT_EQ(gdim, 100);
    ASSERT_GE(bdim, 32);
#else
    ASSERT_EQ(gdim, 1);
    ASSERT_EQ(bdim, 1);
#endif
}

TEST_F(ParallelForTest, InvalidConfigShouldFail)
{
    Zero();
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
        try {
            runParallelForEP(3, ep, 1000, dCount, dGdim, dBdim);
#ifdef HEMI_CUDA_COMPILER
            ASSERT_FAILURE(checkCudaErrors());
#endif
        }
        catch (...) {
#ifdef HEMI_CUDA_COMPILER
            ASSERT_SUCCESS(checkCudaErrors());
#endif
        }

	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
        try {
            runParallelForEP(4, ep, 1000, dCount, dGdim, dBdim);
#ifdef HEMI_CUDA_COMPILER
            ASSERT_FAILURE(checkCudaErrors());
#endif
        }
        catch (...) {
#ifdef HEMI_CUDA_COMPILER
            ASSERT_SUCCESS(checkCudaErrors());
#endif
        }

}
