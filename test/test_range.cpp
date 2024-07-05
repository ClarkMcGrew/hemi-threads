#include "gtest/gtest.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"

#if defined(HEMI_THREADS_DISABLE) && defined(HEMI_CUDA_DISABLE)
#define RangeTest RangeSingle
#elif defined(HEMI_CUDA_DISABLE)
#define Range RangeHost
#else
#define Range RangeDevice
#endif


TEST(RangeTest, SequentialRange) {
    int low = 0;
    int high = 10;
    int step = 1;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high).step(step)) {
        ASSERT_GE(val,low) << "below low end of range";
        ASSERT_LT(val,high) << "above high end of range";
        EXPECT_EQ(expected,val);
        expected += step;
        ++iterations;
    }
    ASSERT_EQ(iterations,(high-low)) << "correct number of iterations";
}

TEST(RangeTest, SteppedRange) {
    int low = 0;
    int high = 10;
    int step = 3;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high).step(step)) {
        ASSERT_GE(val,low) << "below low end of range";
        ASSERT_LT(val,high) << "above high end of range";
        EXPECT_EQ(expected,val);
        expected += step;
        ++iterations;
    }
    ASSERT_EQ(iterations,1+(high-low)/step) << "correct number of iterations";
}

TEST(RangeTest, LimitedRange) {
    int low = 3;
    int high = 13;
    int step = 1;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high).step(step)) {
        ASSERT_GE(val,low) << "below low end of range";
        ASSERT_LT(val,high) << "above high end of range";
        EXPECT_EQ(expected,val);
        expected += step;
        ++iterations;
    }
    ASSERT_EQ(iterations,(high-low)) << "correct number of iterations";
}

TEST(RangeTest, SingleRange) {
    int low = 0;
    int high = 1;
    int step = 1;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high).step(step)) {
        ASSERT_GE(val,low) << "below low end of range";
        ASSERT_LT(val,high) << "above high end of range";
        EXPECT_EQ(expected,val);
        expected += step;
        ++iterations;
    }
    ASSERT_EQ(iterations,(high-low)) << "correct number of iterations";
}

TEST(RangeTest, InvalidRange) {
    int low = 13;
    int high = 3;
    int step = 1;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high)) {
        ++iterations;
        break;
    }
    ASSERT_EQ(iterations,1);
}

TEST(RangeTest, ZeroSteppedRange) {
    int low = 13;
    int high = 3;
    int step = 1;
    int expected = low;
    int iterations = 0;
    for (auto val: util::lang::range(low,high).step(step)) {
        ASSERT_GE(val,low) << "below low end of range";
        ASSERT_LT(val,high) << "above high end of range";
        EXPECT_EQ(expected,val);
        expected += step;
        ++iterations;
    }
}
