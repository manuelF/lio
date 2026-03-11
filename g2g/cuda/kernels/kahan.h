
/**
 * @file kahan.h
 * @brief Reduces FP accumulation error from O(N*eps) to O(eps), making results
 * independent of accumulation order. Each accumulator carries a compensation
 * term that captures the rounding error from each addition.
 */

#ifndef _KAHAN_H_
#define _KAHAN_H_

/**
 * @brief Kahan compensated addition for a scalar.
 * @param sum  Running sum (updated in place).
 * @param comp Compensation term (updated in place, init to 0).
 * @param val  Value to add.
 */
template <typename T>
__device__ __forceinline__ void kahanAdd(T& sum, T& comp, T val) {
  T y = val - comp;
  T t = sum + y;
  comp = (t - sum) - y;
  sum = t;
}

/**
 * @brief Kahan compensated addition for a 3D vector.
 */
template <typename T>
__device__ __forceinline__ void kahanAdd3(vec_type<T, 3>& sum,
                                          vec_type<T, 3>& comp,
                                          vec_type<T, 3> val) {
  kahanAdd(sum.x, comp.x, val.x);
  kahanAdd(sum.y, comp.y, val.y);
  kahanAdd(sum.z, comp.z, val.z);
}

/**
 * @brief Kahan compensated addition for a 4D vector.
 */
template <typename T, unsigned int N>
__device__ __forceinline__ void kahanAdd4(vec_type<T, N>& sum,
                                          vec_type<T, N>& comp,
                                          vec_type<T, N> val) {
  kahanAdd(sum.x, comp.x, val.x);
  kahanAdd(sum.y, comp.y, val.y);
  kahanAdd(sum.z, comp.z, val.z);
  kahanAdd(sum.w, comp.w, val.w);
}

#endif  // _KAHAN_H_