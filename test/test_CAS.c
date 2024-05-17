#include <stdio.h>
#include <stdatomic.h>
#include "test_CAS.h"

int test_CAS_double(double *variable, double *expected, double update) {
     // This is a C11 function to do the atomic CAS.  There is a recent C++
     // interface, but it does not work for non-atomic variables without some
     // fancy footwork in the call.  This is an easier way for people to
     // understand.
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#warning test_CAS -- Using C11 atomic_compare_exchange_weak_explicit
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
#warning test_CAS -- Using builtin __atomic_compare_exchange
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
#warning test_CAS -- Using GCC >5 builtin __atomic_compare_exchange
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
#error Atomic compare and exchange are not supported by this compiler version
     return true;
#endif
}

// See documentation for test_CAS_double
int test_CAS_float(float *variable, float *expected, float update) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
     return true;
#endif
}

// See documentation for test_CAS_double
int test_CAS_int(int *variable, int *expected, int update) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
     return __atomic_compare_exchange(
          variable, expected, update,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
     return true;
#endif
}
