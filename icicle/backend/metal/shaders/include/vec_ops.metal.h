#pragma once

#include <metal_stdlib>
using namespace metal;

/*============================== Element-wise operations =============================*/

template <typename T>
kernel void vector_add(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& size [[buffer(3)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= size) { return; }
  result[tid] = a[tid] + b[tid];
}

template <typename T>
kernel void vector_accumulate(
  device T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  constant uint64_t& size [[buffer(2)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= size) { return; }
  a[tid] = a[tid] + b[tid];
}

template <typename T>
kernel void vector_sub(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& size [[buffer(3)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= size) { return; }
  result[tid] = a[tid] - b[tid];
}

template <typename T>
kernel void vector_mul(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& size [[buffer(3)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= size) { return; }
  result[tid] = a[tid] * b[tid];
}

template <typename T>
kernel void vector_div(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& size [[buffer(3)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= size) { return; }
  result[tid] = a[tid] / b[tid];
}

/*============================== Montgomery conversions  ==============================*/

template <typename T>
kernel void convert_montgomery(
  device const T* input [[buffer(0)]],   // Input buffer
  device T* output [[buffer(1)]],        // Output buffer
  constant uint64_t& size [[buffer(2)]], // Size of the input
  constant bool& is_into [[buffer(3)]],  // Direction of conversion
  uint tid [[thread_position_in_grid]])
{
  // Perform the computation if within bounds
  if (tid >= size) { return; }
  output[tid] = is_into ? T::to_montgomery(input[tid]) : T::from_montgomery(input[tid]);
}

/*============================== Scalar vector operations ==============================*/

template <typename T>
kernel void scalar_add(
  device const T* scalar_a [[buffer(0)]],
  device const T* vec_b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& vec_size [[buffer(3)]],
  constant uint64_t& batch_size [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= vec_size * batch_size) { return; }
  const T y = scalar_a[columns_batch ? tid % batch_size : tid / vec_size];
  result[tid] = vec_b[tid] + y;
}

template <typename T>
kernel void scalar_sub(
  device const T* scalar_a [[buffer(0)]],
  device const T* vec_b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& vec_size [[buffer(3)]],
  constant uint64_t& batch_size [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= vec_size * batch_size) { return; }
  result[tid] = scalar_a[columns_batch ? tid % batch_size : tid / vec_size] - vec_b[tid];
}

template <typename T>
kernel void scalar_mul(
  device const T* scalar_a [[buffer(0)]],
  device const T* vec_b [[buffer(1)]],
  device T* result [[buffer(2)]],
  constant uint64_t& vec_size [[buffer(3)]],
  constant uint64_t& batch_size [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  // Skip execution if tid is out of bounds
  if (tid >= vec_size * batch_size) { return; }
  result[tid] = scalar_a[columns_batch ? tid % batch_size : tid / vec_size] * vec_b[tid];
}

/*============================== Polynomial evaluation  ==============================*/

template <typename T>
kernel void poly_eval(
  device const T* coeffs [[buffer(0)]],
  device const T* domain [[buffer(1)]],
  device T* evals [[buffer(2)]],
  constant uint64_t& coeffs_size [[buffer(3)]],
  constant uint64_t& batch_size [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  constant uint64_t& domain_size [[buffer(6)]],
  uint BlockDim [[threads_per_threadgroup]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint ThreadIdx [[thread_position_in_threadgroup]])
{
  // thread corresponds to idx_in_batch
  uint tid = blockIdx * BlockDim + ThreadIdx;

  // Skip execution if tid is out of bounds
  if (tid >= batch_size * domain_size) { return; }
  uint64_t stride = columns_batch ? batch_size : 1;

  uint idx_in_batch;
  uint idx_in_domain;

  // evals() layout depends on this condition
  if (columns_batch) {
    idx_in_batch = tid % batch_size;
    idx_in_domain = tid / batch_size;
  } else {
    idx_in_batch = tid / domain_size;
    idx_in_domain = tid % domain_size;
  }
  device const T* curr_coeffs = columns_batch ? coeffs + idx_in_batch : coeffs + idx_in_batch * coeffs_size;
  evals[tid] = curr_coeffs[(coeffs_size - 1) * stride];
  // Copy the device value to a thread variable
  T domain_value = domain[idx_in_domain];
  for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
    // Copy the device value to a thread variable
    T curr_coeff_value = curr_coeffs[coeff_idx * stride];
    evals[tid] = evals[tid] * domain_value + curr_coeff_value;
  }
}

/*============================== Polynomial division  ==============================*/

template <typename T>
kernel void school_book_division(
  device T* r [[buffer(0)]],
  device T* q [[buffer(1)]],
  const device T* b [[buffer(2)]],
  const device uint64_t& deg_r [[buffer(3)]],
  const device uint64_t& deg_b [[buffer(4)]],
  uint tid [[thread_position_in_threadgroup]],
  uint block_size [[threads_per_threadgroup]])
{
  threadgroup uint64_t deg_r_shared = 0;
  if (tid == 0) { deg_r_shared = deg_r; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T lc_b_inv = T::inverse(b[deg_b]);
  while (deg_r_shared >= deg_b) {
    int64_t monomial = deg_r_shared - deg_b;
    T lc_r = r[deg_r_shared];
    T monomial_coeff = lc_r * lc_b_inv;

    for (unsigned int i = tid; i <= deg_r_shared - monomial; i += block_size) {
      int global_idx = i + monomial;
      T b_coeff = b[i];
      r[global_idx] = r[global_idx] - monomial_coeff * b_coeff;
    }

    if (tid == 0) {
      q[monomial] = monomial_coeff;
      deg_r_shared--;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

/*============================== Reduction operations for vectors ==============================*/

template <typename T>
struct ReduceAddOp {
  static T neutral() { return T::zero(); }
  static T apply(const thread T& lhs, const thread T& rhs) { return lhs + rhs; }
};

template <typename T>
struct ReduceMulOp {
  static T neutral() { return T::one(); }
  static T apply(const thread T& lhs, const thread T& rhs) { return lhs * rhs; }
};

// Each threadgroup corresponds to exactly one vector. If you dispatch
// `numVectors` threadgroups, then `groupId.x` indicates which vector
// is being reduced.
//
// Requires:
//   - T: data type (e.g., float)
//   - GROUP_SIZE: compile-time constant for threadgroup size
//
// You must ensure that the actual dispatch is consistent with GROUP_SIZE
// (e.g., dispatching 256 threads in x-dimension).

template <typename T, uint GROUP_SIZE, typename Op>
kernel void vector_reduce(
  const device T* input [[buffer(0)]],
  device T* output [[buffer(1)]],
  constant uint64_t& sizeVec [[buffer(2)]],
  constant uint64_t& numVectors [[buffer(3)]],
  constant bool& columns_batch [[buffer(4)]],
  uint groupId [[threadgroup_position_in_grid]],
  uint tid [[thread_position_in_threadgroup]])
{
  // Which vector does this threadgroup handle?
  uint v = groupId;
  if (v >= numVectors) {
    // If the grid has more threadgroups than needed, do nothing.
    return;
  }

  // Stride-based partial sum for the vector v
  T localRes = Op::neutral();

  // We assume the actual # of threads in this group is GROUP_SIZE in x.
  for (uint i = tid; i < sizeVec; i += GROUP_SIZE) {
    uint64_t idx = columns_batch ? i * numVectors + v : v * sizeVec + i;
    T x = input[idx];
    localRes = Op::apply(localRes, x);
  }

  // We need a shared array of size GROUP_SIZE for partial sums.
  // Because MSL requires a compile-time constant for array length,
  // we rely on GROUP_SIZE being a template parameter.
  threadgroup T sharedData[GROUP_SIZE];
  sharedData[tid] = localRes;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Tree reduction in shared memory
  for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      T sd0 = sharedData[tid];
      T sd1 = sharedData[tid + offset];
      sharedData[tid] = Op::apply(sd0, sd1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final result to output[v]
  if (tid == 0) { output[v] = sharedData[0]; }
}

/*============================== Reduction operations for indices ==============================*/

// At this moment, we only support highest_non_zero_idx
// This implementation can support other index reduction operations
// through the use of custom Op struct like below
//
// highest non-zero index reduction operation
// template <typename T>
// struct ReduceHighestNZIdxOp {
//   static int64_t neutral() { return -1; }
//   static int64_t apply(
//     const thread T& state_x, const thread int64_t& state_idx, const thread T& test_x, const thread int64_t& test_idx)
//   {
//     if (test_x > T::zero()) {
//       return test_idx;
//     } else {
//       return state_idx;
//     }
//   }
// };

template <typename T, uint GROUP_SIZE /*, typename Op*/>
kernel void vector_reduce_idx(
  const device T* input [[buffer(0)]],
  device int64_t* out_idx [[buffer(1)]],
  constant uint64_t& sizeVec [[buffer(2)]],
  constant uint64_t& numVectors [[buffer(3)]],
  constant bool& columns_batch [[buffer(4)]],
  uint groupId [[threadgroup_position_in_grid]],
  uint tid [[thread_position_in_threadgroup]])
{
  // Which vector does this threadgroup handle?
  uint v = groupId;
  if (v >= numVectors) {
    // If the grid has more threadgroups than needed, do nothing.
    return;
  }

  // Stride-based partial idx for the vector v
  // int64_t localRes = Op::neutral();
  int64_t localRes = -1;

  // We assume the actual # of threads in this group is GROUP_SIZE in x.
  for (uint i = tid; i < sizeVec; i += GROUP_SIZE) {
    uint64_t idx = columns_batch ? i * numVectors + v : v * sizeVec + i;
    T x = input[idx];
    // localRes = Op::apply(localRes, x);
    if (!(x == T::zero())) { localRes = i; }
  }

  // We need a shared array of size GROUP_SIZE for partial sums.
  // Because MSL requires a compile-time constant for array length,
  // we rely on GROUP_SIZE being a template parameter.
  threadgroup int64_t sharedData[GROUP_SIZE];
  sharedData[tid] = localRes;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Tree reduction in shared memory
  for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      int64_t idx0 = sharedData[tid];
      int64_t idx1 = sharedData[tid + offset];
      // sharedData[tid] = Op::apply(sd0, sd1);
      sharedData[tid] = idx0 > idx1 ? idx0 : idx1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final result to output[v]
  if (tid == 0) { out_idx[v] = sharedData[0]; }
}

/*============================== Slicing ==============================*/

template <typename T>
kernel void slice(
  const device T* in [[buffer(0)]],
  device T* out [[buffer(1)]],
  constant uint64_t& offset [[buffer(2)]],
  constant uint64_t& stride [[buffer(3)]],
  constant uint64_t& in_size [[buffer(4)]],
  constant uint64_t& out_size [[buffer(5)]],
  constant uint64_t& batch_size [[buffer(6)]],
  constant bool& columns_batch [[buffer(7)]],
  uint tid [[thread_position_in_grid]])
{
  if (tid < out_size * batch_size) {
    if (columns_batch) {
      uint64_t vec_num = tid % batch_size;
      uint64_t vec_tid = tid / batch_size;
      out[tid] = in[vec_num + (offset + vec_tid * stride) * batch_size]; // coalesced read and write
    } else {
      uint64_t vec_num = tid / out_size;
      uint64_t vec_tid = tid % out_size;
      out[tid] = in[vec_num * in_size + offset + vec_tid * stride]; // uncoalesced read, coalesced write
    }
  }
}

/*============================== Transpose  ==============================*/

template <typename T>
kernel void matrix_transpose(
  const device T* in [[buffer(0)]],
  device T* out [[buffer(1)]],
  constant uint32_t& row_size [[buffer(2)]],
  constant uint32_t& column_size [[buffer(3)]],
  constant uint64_t& batch_size [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  // Check if the thread is out of bounds
  if (tid < row_size * column_size * batch_size) {
    if (columns_batch) {
      uint64_t vec_num = tid % batch_size;
      uint64_t vec_tid = tid / batch_size;
      // coalesced read and write
      // TODO: in CUDA was:
      // out[vec_num + ((vec_tid % row_size) * column_size + (vec_tid / row_size)) * batch_size] = in[tid];
      out[vec_num + ((vec_tid % column_size) * row_size + (vec_tid / column_size)) * batch_size] = in[tid];
    } else {
      // Calculate the size of each vector (matrix)
      uint64_t vec_size = row_size * column_size;
      // Determine which vector (matrix) in the batch this thread belongs to
      uint64_t vec_num = tid / vec_size;
      // Determine the position within the vector (matrix)
      uint64_t vec_tid = tid % vec_size;
      // Perform the transpose operation
      // Coalesced read, uncoalesced write
      // TODO: in CUDA was:
      // out[vec_num * vec_size + (vec_tid % row_size) * column_size + (vec_tid / row_size)] = in[tid];
      out[vec_num * vec_size + (vec_tid % column_size) * row_size + (vec_tid / column_size)] = in[tid];
    }
  }
}

/*============================== Bit-reverse ==============================*/

uint64_t bit_reverse64(uint64_t x)
{
  x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
  x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
  x = ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4);
  x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8);
  x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16);
  x = ((x >> 32) & 0x00000000FFFFFFFF) | ((x & 0x00000000FFFFFFFF) << 32);
  return x;
}

template <typename T>
kernel void bit_reverse(
  const device T* input [[buffer(0)]],
  device T* output [[buffer(1)]],
  constant uint64_t& vec_size [[buffer(2)]],
  constant uint64_t& batch_size [[buffer(3)]],
  constant uint64_t& shift [[buffer(4)]],
  constant bool& columns_batch [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  // Check if the thread is within the valid range
  if (tid < vec_size * batch_size) {
    if (columns_batch) {
      uint64_t vec_tid = tid / batch_size;
      uint64_t vec_num = tid % batch_size;
      uint64_t reversed_index = bit_reverse64(vec_tid) >> shift;
      output[vec_num + reversed_index * batch_size] = input[tid];
    } else {
      uint64_t vec_num = tid / vec_size;
      uint64_t vec_tid = tid % vec_size;
      uint64_t reversed_index = bit_reverse64(vec_tid) >> shift;
      output[vec_num * vec_size + reversed_index] = input[tid];
    }
  }
}

template <typename T>
kernel void bit_reverse_in_place(
  device T* input [[buffer(0)]],
  constant uint64_t& vec_size [[buffer(1)]],
  constant uint64_t& batch_size [[buffer(2)]],
  constant uint64_t& shift [[buffer(3)]],
  constant bool& columns_batch [[buffer(4)]],
  uint tid [[thread_position_in_grid]])
{
  // Check if the thread is within the valid range
  if (tid < vec_size * batch_size) {
    if (columns_batch) {
      uint64_t vec_tid = tid / batch_size;
      uint64_t vec_num = tid % batch_size;
      uint64_t reversed_index = bit_reverse64(vec_tid) >> shift;
      if (reversed_index > vec_tid) {
        T temp = input[tid];
        input[tid] = input[vec_num + reversed_index * batch_size];
        input[vec_num + reversed_index * batch_size] = temp;
      }
    } else {
      uint64_t vec_num = tid / vec_size;
      uint64_t vec_tid = tid % vec_size;
      uint64_t reversed_index = bit_reverse64(vec_tid) >> shift;
      if (reversed_index > vec_tid) {
        T temp = input[tid];
        input[tid] = input[vec_num * vec_size + reversed_index];
        input[vec_num * vec_size + reversed_index] = temp;
      }
    }
  }
}