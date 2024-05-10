#include "hemi_test.h"
#include "hemi/array.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include <algorithm>

TEST(ArrayTest, CreatesAndFillsArrayOnHost)
{
	const int n = 50;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	float *ptr = data.writeOnlyHostPtr();
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

void fillOnDevice(float* ptr, int n, float val) {
    HEMIFill fillArray;
    hemi::launch(fillArray,ptr,n,val);
}

TEST(ArrayTest, CreatesAndFillsArrayOnDevice)
{
	const int n = 50;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	fillOnDevice(data.writeOnlyPtr(), n, val);

	for(int i = 0; i < n; i++) {
		ASSERT_EQ(val, data.readOnlyPtr(hemi::host)[i]);
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

void squareOnDevice(hemi::Array<float> &a) {
        HEMISquare squareArray;
        hemi::launch(squareArray,a.ptr(),a.size());
}

TEST(ArrayTest, FillsOnHostModifiesOnDevice)
{
	const int n = 50;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	float *ptr = data.writeOnlyHostPtr();
	std::fill(ptr, ptr+n, val);

	squareOnDevice(data);

	for(int i = 0; i < n; i++) {
		ASSERT_EQ(val*val, data.readOnlyPtr(hemi::host)[i]);
	}

	ASSERT_SUCCESS(hemi::deviceSynchronize());
}
