#include <cuda.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <thrust/device_vector.h>

#include "icicle/program/program.h"
#include "icicle/program/symbol.h"
#include "icicle/errors.h"
#include "icicle/backend/vec_ops_backend.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "gpu-utils/utils.h"
#include "cuda_vec_ops.cuh"

#include "icicle/fields/field_config.h"
using namespace field_config;

#define MAX_THREADS_PER_BLOCK 256

template <typename E, typename F, void (*Kernel)(const E*, const F*, uint64_t, E*)>
cudaError_t vec_op(
  const E* a, const F* b, uint64_t size_a, uint64_t size_b, const VecOpsConfig& config, E* result, uint64_t size_res)
{
  CHK_INIT_IF_RETURN();

  size_a *= config.batch_size;
  size_b *= config.batch_size;
  size_res *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  a = config.is_a_on_device ? a : allocate_and_copy_to_device(a, size_a * sizeof(E), cuda_stream);
  b = config.is_b_on_device ? b : allocate_and_copy_to_device(b, size_b * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(size_res * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (size_res + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(a, b, size_res, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, size_res * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)a, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)b, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E, void (*Kernel)(const E*, const E*, uint64_t, uint64_t, bool, E*)>
cudaError_t vec_scalar_op(
  const E* scalar, const E* vec, uint64_t vec_size, const VecOpsConfig& config, E* result, uint64_t res_size)
{
  CHK_INIT_IF_RETURN();

  res_size *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  scalar =
    config.is_a_on_device ? scalar : allocate_and_copy_to_device(scalar, config.batch_size * sizeof(E), cuda_stream);
  vec = config.is_b_on_device ? vec
                              : allocate_and_copy_to_device(vec, vec_size * config.batch_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(res_size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (res_size + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    scalar, vec, vec_size, config.batch_size, config.columns_batch, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, res_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)scalar, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)vec, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

/*============================== add ==============================*/
template <typename E>
__global__ void add_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
}

template <typename E>
eIcicleError
add_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, add_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== inverse ==============================*/
template <typename E>
__global__ void inv_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid].inverse(); }
}

template <typename E>
eIcicleError inv_cuda(const Device& device, const E* vec_a, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, inv_kernel>(vec_a, vec_a, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== accumulate ==============================*/
template <typename E>
eIcicleError accumulate_cuda(const Device& device, E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config)
{
  cudaError_t err = vec_op<E, E, add_kernel>(vec_a, vec_b, size, size, config, vec_a, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void add_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = element_vec[tid] + scalar_val;
  }
}

template <typename E>
eIcicleError add_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, add_scalar_kernel>(scalar_a, vec_b, size, config, result, size);
  return translateCudaError(err);
}

/*============================== sub ==============================*/
template <typename E>
__global__ void sub_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
}

template <typename E>
eIcicleError
sub_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, sub_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void sub_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = scalar_val - element_vec[tid];
  }
}

template <typename E>
eIcicleError sub_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, sub_scalar_kernel>(scalar_a, vec_b, size, config, result, size);
  return translateCudaError(err);
}

/*============================== mul ==============================*/
template <typename E, typename F>
__global__ void mul_kernel(const E* vec_a, const F* vec_b, uint64_t size, E* result)
{
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) { result[tid] = vec_a[tid] * vec_b[tid]; }
}

template <typename E, typename F>
eIcicleError
mul_cuda(const Device& device, const E* vec_a, const F* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, F, mul_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void mul_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = element_vec[tid] * scalar_val;
  }
}

template <typename E>
eIcicleError mul_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, mul_scalar_kernel>(scalar_a, vec_b, size, config, result, size);
  return translateCudaError(err);
}

/*============================== div ==============================*/
template <typename E>
__global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  // TODO:implement better based on https://eprint.iacr.org/2008/199
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] * element_vec2[tid].inverse(); }
}

template <typename E>
eIcicleError
div_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, div_element_wise_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== reduce ==============================*/

struct CustomAdd {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return a + b;
  }
};

struct CustomMul {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return a * b;
  }
};

template <typename E>
cudaError_t
run_reduce(const E* input, uint64_t vec_size, uint64_t batch_size, bool is_sum, E* result, cudaStream_t cuda_stream)
{
  CustomAdd add_op;
  CustomMul mul_op;

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  if (is_sum) {
    CHK_IF_RETURN(cub::DeviceReduce::Reduce(
      d_temp_storage, temp_storage_bytes, input, result, vec_size, add_op, is_sum ? E::zero() : E::one(), cuda_stream));

    // Allocate temporary storage
    CHK_IF_RETURN(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

    // Run reduction
    for (uint64_t i = 0; i < batch_size; i++) {
      CHK_IF_RETURN(cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, input + i * vec_size, result + i, vec_size, add_op,
        is_sum ? E::zero() : E::one(), cuda_stream));
    }
  } else {
    CHK_IF_RETURN(cub::DeviceReduce::Reduce(
      d_temp_storage, temp_storage_bytes, input, result, vec_size, mul_op, is_sum ? E::zero() : E::one(), cuda_stream));

    // Allocate temporary storage
    CHK_IF_RETURN(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

    // Run reduction
    for (uint64_t i = 0; i < batch_size; i++) {
      CHK_IF_RETURN(cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, input + i * vec_size, result + i, vec_size, mul_op,
        is_sum ? E::zero() : E::one(), cuda_stream));
    }
  }
  CHK_IF_RETURN(cudaFreeAsync(d_temp_storage, cuda_stream));
  return CHK_LAST();
}

template <typename E>
cudaError_t reduction(const E* vec_a, uint64_t vec_size, bool is_sum, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  const E* d_vec_a = config.is_a_on_device
                       ? vec_a
                       : allocate_and_copy_to_device(vec_a, vec_size * config.batch_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(config.batch_size * sizeof(E), cuda_stream);

  E* d_transposed;
  if (config.columns_batch) {
    CHK_IF_RETURN(cudaMallocAsync(&d_transposed, vec_size * config.batch_size * sizeof(E), cuda_stream));
    uint64_t num_threads = MAX_THREADS_PER_BLOCK;
    uint64_t num_blocks = (vec_size * config.batch_size + num_threads - 1) / num_threads;
    transpose_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_vec_a, d_transposed, config.batch_size, vec_size, 1);
  }
  const E* d_input = config.columns_batch ? d_transposed : d_vec_a;

  CHK_IF_RETURN(run_reduce(d_input, vec_size, config.batch_size, is_sum, d_result, cuda_stream));

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(
      cudaMemcpyAsync(result, d_result, config.batch_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)d_vec_a, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError
sum_reduce_cuda(const Device& device, const E* vec_a, uint64_t vec_size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = reduction<E>(vec_a, vec_size, true, config, result);
  return translateCudaError(err);
}

template <typename E>
eIcicleError
mul_reduce_cuda(const Device& device, const E* vec_a, uint64_t vec_size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = reduction<E>(vec_a, vec_size, false, config, result);
  return translateCudaError(err);
}

/*============================== Bit-reverse ==============================*/

template <typename E>
__global__ void bit_reverse_kernel(const E* input, uint64_t vec_size, uint64_t batch_size, uint64_t shift, E* output)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < vec_size * batch_size) {
    uint64_t vec_num = tid / vec_size;
    uint64_t vec_tid = tid % vec_size;
    uint64_t reversed_index = __brevll(vec_tid) >> shift;
    output[vec_num * vec_size + reversed_index] = input[tid];
  }
}

template <typename E>
__global__ void
bit_reverse_columns_batch_kernel(const E* input, uint64_t vec_size, uint64_t batch_size, uint64_t shift, E* output)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < vec_size * batch_size) {
    uint64_t vec_tid = tid / batch_size;
    uint64_t vec_num = tid % batch_size;
    uint64_t reversed_index = __brevll(vec_tid) >> shift;
    output[vec_num + reversed_index * batch_size] = input[tid];
  }
}

template <typename E>
__global__ void bit_reverse_inplace_kernel(E* input, uint64_t vec_size, uint64_t batch_size, uint64_t shift)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < vec_size * batch_size) {
    uint64_t vec_num = tid / vec_size;
    uint64_t vec_tid = tid % vec_size;
    uint64_t reversed_index = __brevll(vec_tid) >> shift;
    if (reversed_index > vec_tid) {
      E temp = input[tid];
      input[tid] = input[vec_num * vec_size + reversed_index];
      input[vec_num * vec_size + reversed_index] = temp;
    }
  }
}

template <typename E>
__global__ void
bit_reverse_inplace_columns_batch_kernel(E* input, uint64_t vec_size, uint64_t batch_size, uint64_t shift)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < vec_size * batch_size) {
    uint64_t vec_tid = tid / batch_size;
    uint64_t vec_num = tid % batch_size;
    uint64_t reversed_index = __brevll(vec_tid) >> shift;
    if (reversed_index > vec_tid) {
      E temp = input[tid];
      input[tid] = input[vec_num + reversed_index * batch_size];
      input[vec_num + reversed_index * batch_size] = temp;
    }
  }
}

template <typename E>
cudaError_t bit_reverse_cuda_impl(const E* input, uint64_t size, const VecOpsConfig& cfg, E* output)
{
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(cfg.stream);
  uint64_t total_size = size * cfg.batch_size;

  if (size & (size - 1)) THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "bit_reverse: size must be a power of 2");
  if ((input == output) & (cfg.is_a_on_device != cfg.is_result_on_device))
    THROW_ICICLE_ERR(
      eIcicleError::INVALID_ARGUMENT, "bit_reverse: equal devices should have same is_on_device parameters");

  E* d_output;
  if (cfg.is_result_on_device) {
    d_output = output;
  } else {
    // allocate output on gpu
    CHK_IF_RETURN(cudaMallocAsync(&d_output, sizeof(E) * total_size, cuda_stream));
  }

  uint64_t shift = __builtin_clzll(size) + 1;
  uint64_t num_blocks = (total_size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

  if ((input != output) & cfg.is_a_on_device) {
    if (cfg.columns_batch)
      bit_reverse_columns_batch_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(
        input, size, cfg.batch_size, shift, d_output);
    else
      bit_reverse_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(
        input, size, cfg.batch_size, shift, d_output);
  } else {
    if (!cfg.is_a_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(d_output, input, sizeof(E) * total_size, cudaMemcpyHostToDevice, cuda_stream));
    }
    if (cfg.columns_batch)
      bit_reverse_inplace_columns_batch_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(
        d_output, size, cfg.batch_size, shift);
    else
      bit_reverse_inplace_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(
        d_output, size, cfg.batch_size, shift);
  }
  if (!cfg.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, sizeof(E) * total_size, cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_output, cuda_stream));
  }
  if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename T>
eIcicleError bit_reverse_cuda(const Device& device, const T* in, uint64_t size, const VecOpsConfig& config, T* out)
{
  auto err = bit_reverse_cuda_impl<T>(in, size, config, out);
  return translateCudaError(err);
}

/*============================== slice ==============================*/
template <typename T>
__global__ void slice_kernel(
  const T* in, T* out, uint64_t offset, uint64_t stride, uint64_t in_size, uint64_t out_size, uint64_t batch_size)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < out_size * batch_size) {
    uint64_t vec_num = tid / out_size;
    uint64_t vec_tid = tid % out_size;
    out[tid] = in[vec_num * in_size + offset + vec_tid * stride]; // uncoalesced read, coalesced write
  }
}

template <typename T>
__global__ void slice_columns_batch_kernel(
  const T* in, T* out, uint64_t offset, uint64_t stride, uint64_t in_size, uint64_t out_size, uint64_t batch_size)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < out_size * batch_size) {
    uint64_t vec_num = tid % batch_size;
    uint64_t vec_tid = tid / batch_size;
    out[tid] = in[vec_num + (offset + vec_tid * stride) * batch_size]; // coalesced read and write
  }
}

template <typename E>
cudaError_t _slice_cuda(
  const E* vec_a,
  uint64_t offset,
  uint64_t stride,
  uint64_t size_in,
  uint64_t size_out,
  const VecOpsConfig& config,
  E* result)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  // need to copy consecutive memory where output elements reside in vec_a
  const E* d_vec_a = config.is_a_on_device
                       ? vec_a
                       : allocate_and_copy_to_device(vec_a, size_in * config.batch_size * sizeof(E), cuda_stream);
  E* d_result =
    config.is_result_on_device ? result : allocate_on_device<E>(size_out * config.batch_size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (size_out * config.batch_size + num_threads - 1) / num_threads;
  if (config.columns_batch)
    slice_columns_batch_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_vec_a, d_result, offset, stride, size_in, size_out, config.batch_size);
  else
    slice_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_vec_a, d_result, offset, stride, size_in, size_out, config.batch_size);
  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(
      cudaMemcpyAsync(result, d_result, size_out * config.batch_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)d_vec_a, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError slice_cuda(
  const Device& device,
  const E* vec_a,
  uint64_t offset,
  uint64_t stride,
  uint64_t size_in,
  uint64_t size_out,
  const VecOpsConfig& config,
  E* result)
{
  cudaError_t err = _slice_cuda<E>(vec_a, offset, stride, size_in, size_out, config, result);
  return translateCudaError(err);
}

/*============================== highest non-zero idx ==============================*/
template <typename T>
__global__ void highest_non_zero_idx_kernel(
  const T* vec,
  uint64_t vec_size,
  uint64_t batch_size,
  int run_length,
  int nof_runs,
  int64_t* idx) // TODO - make work with uint64
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nof_runs * batch_size) return;
  int vec_num = tid / nof_runs;
  int vec_tid = tid % nof_runs;
  for (int i = vec_tid + (run_length - 1) * nof_runs; i >= vec_tid; i -= nof_runs) {
    if (i >= vec_size) continue;
    if (vec[vec_num * vec_size + i] != T::zero()) { // coalesced read
      idx[tid] = i;                                 // coalesced write
      return;
    }
  }
  idx[tid] = -1; // -1 for all zeros vec
}

template <typename T>
__global__ void highest_non_zero_idx_columns_batch_kernel(
  const T* vec, uint64_t vec_size, uint64_t batch_size, int run_length, int nof_runs, int64_t* idx)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nof_runs * batch_size) return;
  int vec_num = tid % batch_size;
  int vec_tid = tid / batch_size;
  for (int i = vec_tid + (run_length - 1) * nof_runs; i >= vec_tid; i -= nof_runs) {
    if (i >= vec_size) continue;
    if (vec[vec_num + i * batch_size] != T::zero()) { // coalesced read
      idx[tid] = i;                                   // coalesced write
      return;
    }
  }
  idx[tid] = -1; // -1 for all zeros vec
}

template <typename E>
static __global__ void
transpose_kernel(const E* in, E* out, uint32_t row_size, uint32_t column_size, uint64_t batch_size)
{
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= row_size * column_size * batch_size) return;
  uint64_t vec_size = row_size * column_size;
  uint64_t vec_num = tid / vec_size;
  uint64_t vec_tid = tid % vec_size;
  out[vec_num * vec_size + (vec_tid % row_size) * column_size + (vec_tid / row_size)] =
    in[tid]; // coalesced read, uncoalesed write
}

template <typename E>
cudaError_t _highest_non_zero_idx(const E* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)
{
  /* algorithm explanation:
  stage 1 - each thread reads run_length num of elements and finds the highest non zero index. this is written to
  d_temp_idx. stage 2 - we find the maximum index out of all the high indices in d_temp_idx using a device wide CUB
  function.
  */

  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  int run_length = 7; // TODO - find optimal value
  int nof_runs = (size + run_length - 1) / run_length;

  // allocate device memory and copy if input/output not on device already
  // need to copy consecutive memory where output elements reside in vec_a
  input = config.is_a_on_device ? input
                                : allocate_and_copy_to_device(input, size * config.batch_size * sizeof(E), cuda_stream);
  int64_t* d_out_idx = config.is_result_on_device
                         ? out_idx
                         : allocate_on_device<int64_t>(config.batch_size * sizeof(int64_t), cuda_stream);

  int64_t* d_temp_idx = allocate_on_device<int64_t>(nof_runs * config.batch_size * sizeof(int64_t), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (nof_runs * config.batch_size + num_threads - 1) / num_threads;
  if (config.columns_batch)
    highest_non_zero_idx_columns_batch_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      input, size, config.batch_size, run_length, nof_runs, d_temp_idx);
  else
    highest_non_zero_idx_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      input, size, config.batch_size, run_length, nof_runs, d_temp_idx);

  // transpose for columns batch
  int64_t* d_transposed;
  if (config.columns_batch) {
    CHK_IF_RETURN(cudaMallocAsync(&d_transposed, nof_runs * config.batch_size * sizeof(int64_t), cuda_stream));
    num_threads = MAX_THREADS_PER_BLOCK;
    num_blocks = (nof_runs * config.batch_size + num_threads - 1) / num_threads;
    transpose_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_temp_idx, d_transposed, config.batch_size, nof_runs, 1);
  }
  const int64_t* d_input = config.columns_batch ? d_transposed : d_temp_idx;

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CHK_IF_RETURN(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_out_idx, nof_runs, cuda_stream));

  // Allocate temporary storage
  CHK_IF_RETURN(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

  // Run reduction
  for (uint64_t i = 0; i < config.batch_size; i++) {
    CHK_IF_RETURN(cub::DeviceReduce::Max(
      d_temp_storage, temp_storage_bytes, d_input + i * nof_runs, d_out_idx + i, nof_runs, cuda_stream));
  }
  CHK_IF_RETURN(cudaFreeAsync(d_temp_storage, cuda_stream));
  CHK_IF_RETURN(cudaFreeAsync(d_temp_idx, cuda_stream));
  if (config.columns_batch) { CHK_IF_RETURN(cudaFreeAsync(d_transposed, cuda_stream)); }

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(
      cudaMemcpyAsync(out_idx, d_out_idx, config.batch_size * sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_out_idx, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)input, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError highest_non_zero_idx_cuda(
  const Device& device, const E* vec_a, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)
{
  cudaError_t err = _highest_non_zero_idx<E>(vec_a, size, config, out_idx);
  return translateCudaError(err);
}

/*============================== polynomial evaluation ==============================*/

/* algorithm outline:
1. each domain input - copy into a row of length degree.
2. exclusive scan each row to get all the powers using the CUB function.
3. scalar - vec mul each row with coeffs
4. reduce each row (with batch reduction)
same initial rows for each batch
*/

template <typename T>
__global__ void init_powers(const T* domain, uint64_t domain_size, uint64_t coeffs_size, T* results)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= domain_size * coeffs_size) return;
  uint64_t vec_num = tid / coeffs_size;
  uint64_t vec_tid = tid % coeffs_size;
  results[tid] = vec_tid ? domain[vec_num] : T::one();
  // results[tid] = T::one();
}

template <typename T>
cudaError_t generate_powers(T* powers, uint64_t domain_size, uint64_t coeffs_size, cudaStream_t cuda_stream)
{
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CustomMul mul_op;

  CHK_IF_RETURN(cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, powers, powers, mul_op, coeffs_size, cuda_stream));

  // Allocate temporary storage
  CHK_IF_RETURN(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

  // Run scan
  for (uint64_t i = 0; i < domain_size; i++) {
    CHK_IF_RETURN(cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, powers + i * coeffs_size, powers + i * coeffs_size, mul_op, coeffs_size,
      cuda_stream));
  }
  CHK_IF_RETURN(cudaFreeAsync(d_temp_storage, cuda_stream));
  return CHK_LAST();
}

template <typename E>
cudaError_t _poly_eval(
  const E* coeffs,
  uint64_t coeffs_size,
  const E* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  E* evals /*OUT*/)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  coeffs = config.is_a_on_device
             ? coeffs
             : allocate_and_copy_to_device(coeffs, coeffs_size * config.batch_size * sizeof(E), cuda_stream);
  domain = config.is_b_on_device ? domain : allocate_and_copy_to_device(domain, domain_size * sizeof(E), cuda_stream);
  E* d_evals = config.is_result_on_device
                 ? evals
                 : allocate_on_device<E>(domain_size * config.batch_size * sizeof(E), cuda_stream);
  E* d_powers = allocate_on_device<E>(domain_size * coeffs_size * sizeof(E), cuda_stream);
  E* d_temp = allocate_on_device<E>(domain_size * coeffs_size * sizeof(E), cuda_stream);

  // generate powers
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (domain_size * coeffs_size + num_threads - 1) / num_threads;
  init_powers<<<num_blocks, num_threads, 0, cuda_stream>>>(domain, domain_size, coeffs_size, d_powers);
  CHK_IF_RETURN(generate_powers(d_powers, domain_size, coeffs_size, cuda_stream));

  // transpose for columns batch
  E* d_transposed;
  if (config.columns_batch) {
    CHK_IF_RETURN(cudaMallocAsync(&d_transposed, coeffs_size * config.batch_size * sizeof(E), cuda_stream));
    num_threads = MAX_THREADS_PER_BLOCK;
    num_blocks = (coeffs_size * config.batch_size + num_threads - 1) / num_threads;
    transpose_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      coeffs, d_transposed, config.batch_size, coeffs_size, 1);
  }
  const E* d_input = config.columns_batch ? d_transposed : coeffs;

  // multiply coeefs and powers
  for (uint64_t i = 0; i < config.batch_size; i++) {
    num_threads = MAX_THREADS_PER_BLOCK;
    num_blocks = (domain_size * coeffs_size + num_threads - 1) / num_threads;
    mul_scalar_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_input + i * coeffs_size, d_powers, domain_size, coeffs_size, true, d_temp);

    // reduce
    CHK_IF_RETURN(run_reduce(d_temp, coeffs_size, domain_size, true, d_evals + i * domain_size, cuda_stream));
  }

  // transpose results for columns batch
  E* d_transposed_res;
  if (config.columns_batch) {
    CHK_IF_RETURN(cudaMallocAsync(&d_transposed_res, domain_size * config.batch_size * sizeof(E), cuda_stream));
    num_threads = MAX_THREADS_PER_BLOCK;
    num_blocks = (domain_size * config.batch_size + num_threads - 1) / num_threads;
    transpose_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      d_evals, d_transposed_res, domain_size, config.batch_size, 1);
    CHK_IF_RETURN(cudaFreeAsync(d_evals, cuda_stream));
    d_evals = d_transposed_res;
  }

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(
      evals, d_evals, domain_size * config.batch_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_evals, cuda_stream));
  }
  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)coeffs, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)domain, cuda_stream)); }
  if (config.columns_batch) { CHK_IF_RETURN(cudaFreeAsync(d_transposed, cuda_stream)); }
  CHK_IF_RETURN(cudaFreeAsync(d_powers, cuda_stream));
  CHK_IF_RETURN(cudaFreeAsync(d_temp, cuda_stream));
  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename E>
eIcicleError poly_eval_cuda(
  const Device& device,
  const E* coeffs,
  uint64_t coeffs_size,
  const E* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  E* evals /*OUT*/)
{
  cudaError_t err = _poly_eval<E>(coeffs, coeffs_size, domain, domain_size, config, evals);
  return translateCudaError(err);
}

/*============================== polynomial division ==============================*/

template <typename T>
__global__ void school_book_division(T* r, T* q, const T* b, uint64_t deg_r, uint64_t deg_b)
{
  // Assuming single block for intra-block synchronization
  // thread-0 updates deg_r in shared memory
  // all threads are involved in the subtraction part (TODO optimize coalescing)

  if (blockIdx.x > 0) return;
  const uint64_t tid = threadIdx.x;
  const uint32_t block_size = blockDim.x;

  __shared__ uint64_t deg_r_shared;
  if (tid == 0) { deg_r_shared = deg_r; }
  __syncthreads();

  const T lc_b_inv = b[deg_b].inverse();
  while (deg_r_shared >= deg_b) {
    // computing one step 'r = r-sb' (for 'a = q*b+r') where s is a monomial such that 'r-sb' removes the highest degree
    // of r.

    int64_t monomial = deg_r_shared - deg_b; // monomial=1 is 'x', monomial=2 is x^2 etc.
    T lc_r = r[deg_r_shared];
    // Note: skipping when lc_r is zero makes the code slower
    T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b

    for (int i = tid; i <= deg_r_shared - monomial; i += block_size) {
      int global_idx = i + monomial;
      T b_coeff = b[i];
      r[global_idx] = r[global_idx] - monomial_coeff * b_coeff;
    }

    if (tid == 0) {
      q[monomial] = monomial_coeff;
      deg_r_shared--;
    }
    __syncthreads();
  }
}

template <typename T>
cudaError_t _poly_divide_cuda(
  const T* numerator,
  uint64_t numerator_size,
  const T* denominator,
  uint64_t denominator_size,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  uint64_t q_size,
  T* r_out /*OUT*/,
  uint64_t r_size)
{
  CHK_INIT_IF_RETURN();

  auto cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  // copy inputs to device if on host. Note that no need to copy numerator to device since we copy it to r already
  denominator = config.is_b_on_device
                  ? denominator
                  : allocate_and_copy_to_device(denominator, denominator_size * sizeof(T), cuda_stream);

  // compute degree of numerator and denumerator
  int64_t numerator_deg, denominator_deg;
  auto degree_config = default_vec_ops_config();
  degree_config.stream = config.stream;
  degree_config.is_async = config.is_async;
  degree_config.is_a_on_device = config.is_a_on_device; // numerator
  CHK_IF_RETURN(_highest_non_zero_idx(numerator, numerator_size, degree_config, &numerator_deg));
  degree_config.is_a_on_device = config.is_b_on_device; // denominator
  CHK_IF_RETURN(_highest_non_zero_idx(denominator, denominator_size, degree_config, &denominator_deg));

  // verify outputs large enough
  ICICLE_ASSERT(r_size >= (1 + denominator_deg))
    << "polynomial division expects r(x) size to be similar to numerator(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denominator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denominator)+1";

  T* d_r_out = config.is_result_on_device ? r_out : allocate_on_device<T>(r_size * sizeof(T), cuda_stream);
  T* d_q_out = config.is_result_on_device ? q_out : allocate_on_device<T>(q_size * sizeof(T), cuda_stream);

  CHK_IF_RETURN(cudaMemset(d_r_out, 0, r_size * sizeof(T)));
  CHK_IF_RETURN(cudaMemcpyAsync(
    d_r_out, numerator, (1 + numerator_deg) * sizeof(T),
    config.is_a_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cuda_stream));

  degree_config.is_a_on_device = true;
  int64_t deg_r = numerator_deg;
  // NOTE: this kernel works only with a single block due to intra-block synchronization
  school_book_division<<<1, 32 /* Empirical best value*/, 0, cuda_stream>>>(
    d_r_out, d_q_out, denominator, deg_r, denominator_deg);

  // copy back output to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(r_out, d_r_out, r_size * sizeof(T), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaMemcpyAsync(q_out, d_q_out, q_size * sizeof(T), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_r_out, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_q_out, cuda_stream));
  }

  // release device memory, if allocated (skipping numerator since we don't copy it to device)
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)denominator, cuda_stream)); }

  return CHK_LAST();
}

template <typename T>
eIcicleError poly_divide_cuda(
  const Device& device,
  const T* numerator,
  uint64_t numerator_size,
  const T* denominator,
  uint64_t denominator_size,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  uint64_t q_size,
  T* r_out /*OUT*/,
  uint64_t r_size)
{
  if (config.batch_size != 1 && config.columns_batch) {
    ICICLE_LOG_ERROR << "polynomial division is not implemented for column batch. Planned for v3.2";
    return eIcicleError::API_NOT_IMPLEMENTED;
  }

  // TODO v3.2: when implementing batch properly in v3.2, allocate memory once for all batch and use threads instead of
  // a loop
  for (auto batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
    cudaError_t err = _poly_divide_cuda<T>(
      numerator + batch_idx * numerator_size, numerator_size, denominator + batch_idx * denominator_size,
      denominator_size, config, q_out + batch_idx * q_size, q_size, r_out + batch_idx * r_size, r_size);
    auto icicle_err = translateCudaError(err);
    if (icicle_err != eIcicleError::SUCCESS) { return icicle_err; }
  }

  // wait for stream to empty is not async
  auto cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  if (!config.is_async) return translateCudaError(cudaStreamSynchronize(cuda_stream));
  return eIcicleError::SUCCESS;
}

/*============================== program execution ==============================*/
template <typename T>
__global__ void execute_program_kernel(
  T** data,
  const int input_length,
  const InstructionType* instructions,
  int nof_variables,
  int nof_constants,
  int nof_instructions)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= input_length) return;

  execute_instructions<T>(data, instructions, nof_instructions, nof_variables, nof_constants, tid);
}

/*
To determine whether a data's vector resides in host or device the function only config.is_a_on_device is used. It also
determines if the data is copied to the host after the computation is done.
*/
template <typename T>
cudaError_t
program_execution(std::vector<T*>& data, const Program<T>& program, const uint64_t size, const VecOpsConfig& config)
{
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  uint64_t total_size = size * config.batch_size;

  int nof_parameters = program.m_nof_parameters;
  int data_len = nof_parameters + program.m_nof_constants + program.m_nof_intermidiates;

  thrust::device_vector<T*> data_device(data_len);
  int device_data_idx = 0;

  // copy vectors to device.
  for (int idx = 0; idx < nof_parameters; ++idx) {
    // if is_a_on_device is true - copy pointer to thrust's device_vector
    if (config.is_a_on_device) data_device[device_data_idx++] = (data[idx]);
    // if is_a_on_device is false - copy vectors to device
    else
      data_device[device_data_idx++] =
        const_cast<T*>(allocate_and_copy_to_device<T>(data[idx], total_size * sizeof(T), cuda_stream));
  }

  // allocate and copy constants
  for (int idx = 0; idx < program.m_nof_constants; ++idx)
    data_device[device_data_idx++] =
      const_cast<T*>(allocate_and_copy_to_device<T>(&program.m_constants[idx], sizeof(T), cuda_stream));

  // allocate intermidiates
  for (int idx = 0; idx < program.m_nof_intermidiates; ++idx)
    data_device[device_data_idx++] = allocate_on_device<T>(total_size * sizeof(T), cuda_stream);

  // allocate and copy the instructions to device
  const InstructionType* instructions_device = allocate_and_copy_to_device<InstructionType>(
    program.m_instructions.data(), sizeof(InstructionType) * program.m_instructions.size(), cuda_stream);

  // execute kernel
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (total_size + num_threads - 1) / num_threads;
  T** data_device_pointer = thrust::raw_pointer_cast(data_device.data());
  execute_program_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    data_device_pointer, total_size, instructions_device, nof_parameters, program.m_nof_constants,
    program.m_instructions.size());

  // copy back data and release memory if needed
  // copy data
  for (int idx = 0; idx < nof_parameters; ++idx) {
    if (config.is_a_on_device) {
      data[idx] = data_device[idx];
    } else
      CHK_IF_RETURN(
        cudaMemcpyAsync(data[idx], data_device[idx], total_size * sizeof(T), cudaMemcpyDeviceToHost, cuda_stream));
  }

  // release device memory
  if (!config.is_a_on_device) {
    for (int idx = 0; idx < data_device.size(); ++idx) {
      CHK_IF_RETURN(cudaFreeAsync((void*)data_device[idx], cuda_stream));
    }
  } else {
    for (int idx = nof_parameters; idx < data_device.size(); ++idx)
      CHK_IF_RETURN(cudaFreeAsync((void*)data_device[idx], cuda_stream));
  }

  // wait for stream to empty if not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename T>
eIcicleError cuda_execute_program(
  const Device& device,
  std::vector<T*>& data,
  const Program<T>& program,
  const uint64_t size,
  const VecOpsConfig& config)
{
  if (data.size() != program.m_nof_parameters) {
    ICICLE_LOG_ERROR << "Program has " << program.m_nof_parameters << " while data has " << data.size()
                     << " parameters";
    return eIcicleError::INVALID_ARGUMENT;
  }
  cudaError_t err = program_execution(data, program, size, config);
  return translateCudaError(err);
}

/************************************ REGISTRATION ************************************/

REGISTER_VECTOR_ADD_BACKEND("CUDA", add_cuda<scalar_t>);
REGISTER_VECTOR_ACCUMULATE_BACKEND("CUDA", accumulate_cuda<scalar_t>);
REGISTER_VECTOR_INV_BACKEND("CUDA", inv_cuda<scalar_t>);
REGISTER_VECTOR_SUB_BACKEND("CUDA", sub_cuda<scalar_t>);
REGISTER_VECTOR_MUL_BACKEND("CUDA", (mul_cuda<scalar_t, scalar_t>));
REGISTER_VECTOR_SUM_BACKEND("CUDA", sum_reduce_cuda<scalar_t>);
REGISTER_VECTOR_PRODUCT_BACKEND("CUDA", mul_reduce_cuda<scalar_t>);
REGISTER_VECTOR_DIV_BACKEND("CUDA", div_cuda<scalar_t>);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", (mul_scalar_cuda<scalar_t>));
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", (add_scalar_cuda<scalar_t>));
REGISTER_SCALAR_SUB_VEC_BACKEND("CUDA", (sub_scalar_cuda<scalar_t>));
REGISTER_BIT_REVERSE_BACKEND("CUDA", bit_reverse_cuda<scalar_t>);
REGISTER_SLICE_BACKEND("CUDA", slice_cuda<scalar_t>);
REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CUDA", highest_non_zero_idx_cuda<scalar_t>)
REGISTER_POLYNOMIAL_EVAL("CUDA", poly_eval_cuda<scalar_t>);
REGISTER_POLYNOMIAL_DIVISION("CUDA", poly_divide_cuda<scalar_t>);
REGISTER_EXECUTE_PROGRAM_BACKEND("CUDA", cuda_execute_program<scalar_t>);

#ifdef EXT_FIELD
REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("CUDA", add_cuda<extension_t>);
REGISTER_VECTOR_ACCUMULATE_EXT_FIELD_BACKEND("CUDA", accumulate_cuda<extension_t>);
REGISTER_VECTOR_INV_EXT_FIELD_BACKEND("CUDA", inv_cuda<extension_t>);
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("CUDA", sub_cuda<extension_t>);
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("CUDA", (mul_cuda<extension_t, extension_t>));
REGISTER_VECTOR_MIXED_MUL_BACKEND("CUDA", (mul_cuda<extension_t, scalar_t>));
REGISTER_VECTOR_DIV_EXT_FIELD_BACKEND("CUDA", div_cuda<extension_t>);
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND("CUDA", bit_reverse_cuda<extension_t>);
REGISTER_SLICE_EXT_FIELD_BACKEND("CUDA", slice_cuda<extension_t>);
REGISTER_VECTOR_SUM_EXT_FIELD_BACKEND("CUDA", sum_reduce_cuda<extension_t>);
REGISTER_VECTOR_PRODUCT_EXT_FIELD_BACKEND("CUDA", mul_reduce_cuda<extension_t>);
REGISTER_SCALAR_MUL_VEC_EXT_FIELD_BACKEND("CUDA", mul_scalar_cuda<extension_t>);
REGISTER_SCALAR_ADD_VEC_EXT_FIELD_BACKEND("CUDA", add_scalar_cuda<extension_t>);
REGISTER_SCALAR_SUB_VEC_EXT_FIELD_BACKEND("CUDA", sub_scalar_cuda<extension_t>);
REGISTER_EXECUTE_PROGRAM_EXT_FIELD_BACKEND("CUDA", cuda_execute_program<extension_t>);
#endif // EXT_FIELD

#ifdef RING
#include "icicle/rings/integer_ring_rns.h"
REGISTER_VECTOR_ADD_RING_RNS_BACKEND("CUDA", add_cuda<scalar_rns_t>);
REGISTER_VECTOR_ACCUMULATE_RING_RNS_BACKEND("CUDA", accumulate_cuda<scalar_rns_t>);
REGISTER_VECTOR_SUB_RING_RNS_BACKEND("CUDA", sub_cuda<scalar_rns_t>);
REGISTER_VECTOR_MUL_RING_RNS_BACKEND("CUDA", (mul_cuda<scalar_rns_t, scalar_rns_t>));
REGISTER_VECTOR_DIV_RING_RNS_BACKEND("CUDA", div_cuda<scalar_rns_t>);
REGISTER_VECTOR_INV_RING_RNS_BACKEND("CUDA", inv_cuda<scalar_rns_t>);

REGISTER_BIT_REVERSE_RING_RNS_BACKEND("CUDA", bit_reverse_cuda<scalar_rns_t>);
REGISTER_SLICE_RING_RNS_BACKEND("CUDA", slice_cuda<scalar_rns_t>);
REGISTER_VECTOR_SUM_RING_RNS_BACKEND("CUDA", sum_reduce_cuda<scalar_rns_t>);
REGISTER_VECTOR_PRODUCT_RING_RNS_BACKEND("CUDA", mul_reduce_cuda<scalar_rns_t>);
REGISTER_SCALAR_MUL_VEC_RING_RNS_BACKEND("CUDA", mul_scalar_cuda<scalar_rns_t>);
REGISTER_SCALAR_ADD_VEC_RING_RNS_BACKEND("CUDA", add_scalar_cuda<scalar_rns_t>);
REGISTER_SCALAR_SUB_VEC_RING_RNS_BACKEND("CUDA", sub_scalar_cuda<scalar_rns_t>);
REGISTER_EXECUTE_PROGRAM_RING_RNS_BACKEND("CUDA", cuda_execute_program<scalar_rns_t>);

/*============================== RNS conversions ==============================*/
template <typename Zq, typename ZqRns>
__global__ void convert_to_rns_kernel(const Zq* input, uint64_t size, ZqRns* output)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { ZqRns::convert_direct_to_rns(&input[tid].limbs_storage, &output[tid].limbs_storage); }
}

template <typename ZqRns, typename Zq>
__global__ void convert_from_rns_kernel(const ZqRns* input, uint64_t size, Zq* output)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { ZqRns::convert_rns_to_direct(&input[tid].limbs_storage, &output[tid].limbs_storage); }
}

template <typename SrcType, typename DstType, bool convert_into_rns>
eIcicleError
cuda_convert_rns(const Device& device, const SrcType* input, uint64_t size, const VecOpsConfig& config, DstType* output)
{
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  const size_t total_nof_elements = size * config.batch_size;
  const SrcType* d_input = config.is_a_on_device
                             ? input
                             : allocate_and_copy_to_device(input, total_nof_elements * sizeof(SrcType), cuda_stream);
  DstType* d_output =
    config.is_a_on_device ? output : allocate_on_device<DstType>(total_nof_elements * sizeof(DstType), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (total_nof_elements + num_threads - 1) / num_threads;

  if constexpr (convert_into_rns) {
    convert_to_rns_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(d_input, total_nof_elements, d_output);
  } else {
    convert_from_rns_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(d_input, total_nof_elements, d_output);
  }

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    cudaMemcpyAsync(output, d_output, total_nof_elements * sizeof(DstType), cudaMemcpyDeviceToHost, cuda_stream);
    cudaFreeAsync(d_output, cuda_stream);
  }

  // release device memory, if allocated
  if (!config.is_a_on_device) { cudaFreeAsync((void*)d_input, cuda_stream); }
  // wait for stream to empty is not async
  if (!config.is_async) { cudaStreamSynchronize(cuda_stream); }

  return eIcicleError::SUCCESS;
}

REGISTER_CONVERT_TO_RNS_BACKEND("CUDA", (cuda_convert_rns<scalar_t, scalar_rns_t, true /*to_rns*/>));
REGISTER_CONVERT_FROM_RNS_BACKEND("CUDA", (cuda_convert_rns<scalar_rns_t, scalar_t, false /*from_rns*/>));
#endif // RING
