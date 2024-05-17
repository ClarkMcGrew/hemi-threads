#ifndef test_CAS_h_SEEN
#define test_CAS_h_SEEN

// Implement a "portable" atomic compare-and-set for various native types.
// The implentation is usually done using C, and usually leverages the C
// stdatomic definitions, but might depend on compiler built-in methods.  This
// hides the details.  The semantics are similar to the C11
// atomic_compare_exchange_weak_explicit function.  If the value pointed to by
// "variable" is equal to the value pointed to by "expected", then set the
// value pointed to by "variable" to be "update".
#ifdef __cplusplus
extern "C" {
#endif
int test_CAS_double(double *variable, double *expected, double update);
int test_CAS_float(float *variable, float *expected, float update);
int test_CAS_int(int *variable, int *expected, int update);
#ifdef __cplusplus
}
#endif


#endif
