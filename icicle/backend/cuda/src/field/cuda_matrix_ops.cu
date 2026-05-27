#include "icicle/backend/mat_ops_backend.h"
#include "icicle/backend/vec_ops_backend.h"
#include "gpu-utils/utils.h"
#include "error_translation.h"
#include "icicle/fields/field_config.h"
#include <cstdint>

constexpr uint32_t MAX_THREADS_PER_BLOCK = 256;

// CUDA kernel for matrix multiplication
template <typename T, int degree>
__global__ void tqMatmulKernel(
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  T* mat_out,
  bool a_transposed,
  bool b_transposed)
{
  // Calculate effective dimensions based on transpose flags
  uint32_t effective_rows_a = a_transposed ? nof_cols_a : nof_rows_a;
  uint32_t effective_cols_a = a_transposed ? nof_rows_a : nof_cols_a;
  uint32_t effective_cols_b = b_transposed ? nof_rows_b : nof_cols_b;

  // Calculate row and column for this thread
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);

  // Check if thread is within bounds
  if (row < effective_rows_a && col < effective_cols_b) {
    T sum[degree];
    std::memset(sum, 0, sizeof(sum));

    // Compute dot product for this element
    for (uint32_t k = 0; k < effective_cols_a; k++) {
      for (uint32_t l = 0; l < degree; l++) {
        // Adjust indexing for transposed matrix A
        uint32_t idx_a;
        if (a_transposed) {
          // Access A^T: row `row` of A^T is column `row` of A
          idx_a = (k * nof_cols_a + row) * degree + l;
        } else {
          // Access A normally: row `row` of A
          idx_a = (row * nof_cols_a + k) * degree + l;
        }

        // Adjust indexing for transposed matrix B
        uint32_t idx_b;
        if (b_transposed) {
          // Access B^T: column `col` of B^T is row `col` of B
          idx_b = (col * nof_cols_b + k) * degree + l;
        } else {
          // Access B normally: column `col` of B
          idx_b = (k * nof_cols_b + col) * degree + l;
        }

        sum[l] = sum[l] + (mat_a[idx_a] * mat_b[idx_b]);
      }
    }

    auto out = (row * effective_cols_b + col) * degree;
    std::memcpy(mat_out + out, sum, sizeof(T) * degree);
  }
}

template <typename T, uint32_t degree>
static eIcicleError cuda_matmul_internal(
  const Device& device,
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  const MatMulConfig& config,
  T* mat_out)
{
  // Check for null pointers
  if (mat_a == nullptr || mat_b == nullptr || mat_out == nullptr) {
    ICICLE_LOG_ERROR << "Matmul: invalid input pointer";
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Check for zero dimensions
  if (nof_rows_a == 0 || nof_cols_a == 0 || nof_rows_b == 0 || nof_cols_b == 0) {
    ICICLE_LOG_ERROR << "Matmul: invalid input size: size must be >0";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.result_transposed) {
    ICICLE_LOG_ERROR << "[CUDA] Matmul with transposed output is not yet supported. This feature will be available in "
                        "a future release.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Handle transpose flags: if a_transposed is true, we compute A^T * B
  // if b_transposed is true, we compute A * B^T (or A^T * B^T if both are true)
  uint32_t effective_rows_a = config.a_transposed ? nof_cols_a : nof_rows_a;
  uint32_t effective_cols_a = config.a_transposed ? nof_rows_a : nof_cols_a;
  uint32_t effective_rows_b = config.b_transposed ? nof_cols_b : nof_rows_b;
  uint32_t effective_cols_b = config.b_transposed ? nof_rows_b : nof_cols_b;

  // Check if inner dimensions match for matrix multiplication
  if (effective_cols_a != effective_rows_b) {
    ICICLE_LOG_ERROR << "Matmul: inner dimensions do not match (effective_cols_a != effective_rows_b)";
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Get a CUDA stream (use the one from config if provided, otherwise use default)
  cudaStream_t stream = config.stream ? static_cast<cudaStream_t>(config.stream) : 0;

  const size_t size_a = nof_rows_a * nof_cols_a * sizeof(T) * degree;
  const size_t size_b = nof_rows_b * nof_cols_b * sizeof(T) * degree;
  const size_t size_out = effective_rows_a * effective_cols_b * sizeof(T) * degree;

  // Allocate and copy to device memory if inputs are on host memory
  const T* d_mat_a =
    config.is_a_on_device ? mat_a : allocate_and_copy_to_device(mat_a, size_a, stream, config.is_async);
  const T* d_mat_b =
    config.is_b_on_device ? mat_b : allocate_and_copy_to_device(mat_b, size_b, stream, config.is_async);
  T* d_mat_out = config.is_result_on_device ? mat_out : allocate_on_device<T>(size_out, stream, config.is_async);

  // Define kernel launch parameters
  dim3 threadsPerBlock(16, 16); // 16x16 = 256 threads per block
  dim3 numBlocks(
    (effective_cols_b + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (effective_rows_a + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // This implementation currently only support two cases. TODO Lisa: generalize
  static_assert(degree == 64 || degree == 1, "Unsupported degree size.");
  tqMatmulKernel<T, degree><<<numBlocks, threadsPerBlock, 0, stream>>>(
    d_mat_a, nof_rows_a, nof_cols_a, d_mat_b, nof_rows_b, nof_cols_b, d_mat_out, config.a_transposed,
    config.b_transposed);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    cudaMemcpyAsync(mat_out, d_mat_out, size_out, cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_mat_out, stream);
  }

  // release device memory, if allocated
  if (!config.is_a_on_device) { cudaFreeAsync((void*)d_mat_a, stream); }
  if (!config.is_b_on_device) { cudaFreeAsync((void*)d_mat_b, stream); }

  // Synchronize if not async
  if (!config.is_async) { cudaStreamSynchronize(stream); }

  CHK_LAST();

  return eIcicleError::SUCCESS;
}

template <typename T>
static eIcicleError cuda_matmul(
  const Device& device,
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  const MatMulConfig& config,
  T* mat_out)
{
  return cuda_matmul_internal<T, 1>(
    device, mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
}

/**
 * @brief Matrix multiplication for PolyRing<T::Base, T::d>
 *
 * Internally expands each ring element into its base field representation,
 * performs coefficient-wise multiplication, and reassembles the result.
 */
template <typename T>
static eIcicleError cuda_matmul_polynomial_ring(
  const Device& device,
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  const MatMulConfig& config,
  T* mat_out)
{
  using Zq = typename T::Base;
  constexpr uint32_t d = T::d;

  return cuda_matmul_internal<Zq, d>(
    device, reinterpret_cast<const Zq*>(mat_a), nof_rows_a, nof_cols_a, reinterpret_cast<const Zq*>(mat_b), nof_rows_b,
    nof_cols_b, config, reinterpret_cast<Zq*>(mat_out));
}

// === Registration with runtime ===
REGISTER_MATMUL_BACKEND("CUDA", cuda_matmul<field_config::scalar_t>);
#ifdef RING
REGISTER_POLY_RING_MATMUL_BACKEND("CUDA", cuda_matmul_polynomial_ring<field_config::PolyRing>);
#endif

/*============================== transpose ==============================*/

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
cudaError_t
transpose_internal(const E* vec_a, uint32_t row_size, uint32_t column_size, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  uint64_t vec_size = row_size * column_size;
  const E* d_vec_a = config.is_a_on_device
                       ? vec_a
                       : allocate_and_copy_to_device(vec_a, vec_size * config.batch_size * sizeof(E), cuda_stream);
  // offset = config.is_a_on_device ? offset : 0; // already copied from offset so reset to zero
  E* d_result =
    config.is_result_on_device ? result : allocate_on_device<E>(vec_size * config.batch_size * sizeof(E), cuda_stream);

  // If computing inplace using device memory, need to copy input to temporary memory
  const bool is_inplace_with_device_memory =
    config.is_result_on_device && config.is_a_on_device && (d_result == d_vec_a);
  if (is_inplace_with_device_memory) {
    // allocate temporary memory for input
    E* d_temp_input;
    CHK_IF_RETURN(cudaMallocAsync(&d_temp_input, vec_size * config.batch_size * sizeof(E), cuda_stream));
    // copy input to temporary memory
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_temp_input, d_vec_a, vec_size * config.batch_size * sizeof(E), cudaMemcpyDeviceToDevice, cuda_stream));
    d_vec_a = d_temp_input; // use temporary memory for input
  }

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (vec_size * config.batch_size + num_threads - 1) / num_threads;
  transpose_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    d_vec_a, d_result, row_size, column_size, config.batch_size);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(
      cudaMemcpyAsync(result, d_result, vec_size * config.batch_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated or if inplace with device memory (case where we need extra memory)
  if (!config.is_a_on_device || is_inplace_with_device_memory) {
    CHK_IF_RETURN(cudaFreeAsync((void*)d_vec_a, cuda_stream));
  }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError matrix_transpose_cuda(
  const Device& device, const E* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, E* mat_out)
{
  if (!mat_in || !mat_out || nof_rows == 0 || nof_cols == 0) {
    ICICLE_LOG_ERROR << "Matrix-transpose: Invalid pointer or size";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    // What is that even meaning here?
    ICICLE_LOG_ERROR << "Matrix-transpose does not support columns_batch";
    return eIcicleError::INVALID_ARGUMENT;
  }

  cudaError_t err = transpose_internal<E>(mat_in, nof_cols, nof_rows, config, mat_out);
  return translateCudaError(err);
}

REGISTER_MATRIX_TRANSPOSE_BACKEND("CUDA", matrix_transpose_cuda<field_config::scalar_t>);
#ifdef EXT_FIELD
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND("CUDA", matrix_transpose_cuda<field_config::extension_t>);
#endif // EXT_FIELD
#ifdef RING
REGISTER_MATRIX_TRANSPOSE_RING_RNS_BACKEND("CUDA", matrix_transpose_cuda<field_config::scalar_rns_t>);
REGISTER_MATRIX_TRANSPOSE_POLY_RING_BACKEND("CUDA", matrix_transpose_cuda<field_config::PolyRing>);
#endif // RING