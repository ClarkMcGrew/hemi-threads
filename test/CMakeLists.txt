cmake_minimum_required( VERSION 3.14 FATAL_ERROR )
project( hemiThreadsTest )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main  # GoogleTest recommends "living at the head"
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

include( CheckLanguage )
check_language( CUDA )
if( CMAKE_CUDA_COMPILER )
  enable_language(CUDA)
  if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
    # The default is taken from the CUDAARCHS environment
    # variable.  If it isn't set, then set it to the earliest
    # non-deprecated architecture.
    #   2022: architectures before 52 are deprecated.
    if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.23 )
      # After cmake 3.23, this can be set to all or all-major
      set( CMAKE_CUDA_ARCHITECTURES native )
    else()
      set( CMAKE_CUDA_ARCHITECTURES 52 )
    endif()
  endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/..)

enable_testing()

add_executable(test_hemi_single
  test_CAS.c
  test_externals.cpp
  test_execution_policy.cpp
  test_portable_launch.cpp
  test_parallel_for.cpp
  test_array.cpp)
target_compile_definitions(test_hemi_single PUBLIC HEMI_THREADS_DISABLE HEMI_CUDA_DISABLE)
target_link_libraries(test_hemi_single PUBLIC GTest::gtest_main)
gtest_discover_tests(test_hemi_single)

add_executable(test_hemi_host
  test_CAS.c
  test_externals.cpp
  test_range.cpp
  test_execution_policy.cpp
  test_portable_launch.cpp
  test_parallel_for.cpp
  test_array.cpp)
target_compile_definitions(test_hemi_host PUBLIC HEMI_CUDA_DISABLE)
# target_compile_definitions(test_hemi_host PUBLIC HEMI_ARRAY_WITHOUT_MIRROR)
# target_compile_definitions(test_hemi_host PUBLIC HEMI_ARRAY_DEBUG)
# target_compile_definitions(test_hemi_host PUBLIC HEMI_THREADS_DEBUG)
target_link_libraries(test_hemi_host PUBLIC GTest::gtest_main)
gtest_discover_tests(test_hemi_host)

add_executable(test_hemi_device
  test_CAS.c
  test_externals.cpp
  test_execution_policy.cu
  test_launch.cu
  test_portable_launch.cu
  test_parallel_for.cu
  test_array.cu)
target_compile_options(test_hemi_device PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_hemi_device PUBLIC GTest::gtest_main)
target_compile_definitions(test_hemi_device PUBLIC HEMI_CUDA_CHECK_ERROR)
gtest_discover_tests(test_hemi_device)
