#include "Metal/Metal.h"
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "metal_device_api.h"
#include "metal_library_loader.h"
#include "mont_conversion.h"

using namespace field_config;
using namespace icicle;

template <typename T, const char* KERNEL_NAME>
eIcicleError metal_3vectors_op(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  size *= config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device ? metal::map_to_metal_buffer(vec_a)
                                                  : metal::allocate_and_copy_to_device(vec_a, size * sizeof(T));
  auto [bufferB, offsetB] = config.is_b_on_device ? metal::map_to_metal_buffer(vec_b)
                                                  : metal::allocate_and_copy_to_device(vec_b, size * sizeof(T));
  auto [bufferRes, offsetRes] =
    config.is_result_on_device ? metal::map_to_metal_buffer(output) : metal::allocate_on_device(size * sizeof(T));

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferB, offsetB, 1);
  encoder->setBuffer(bufferRes, offsetRes, 2);
  // Bind constants
  encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 3 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(output, bufferRes->contents(), size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }
  if (!config.is_b_on_device) { bufferB->release(); }

  return eIcicleError::SUCCESS;
}

template <typename T, const char* KERNEL_NAME>
eIcicleError
metal_2vectors_op(const Device& device, T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config)
{
  size *= config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device ? metal::map_to_metal_buffer(vec_a)
                                                  : metal::allocate_and_copy_to_device(vec_a, size * sizeof(T));
  auto [bufferB, offsetB] = config.is_b_on_device ? metal::map_to_metal_buffer(vec_b)
                                                  : metal::allocate_and_copy_to_device(vec_b, size * sizeof(T));
  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferB, offsetB, 1);
  // Bind constants
  encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 2 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) { std::memcpy(vec_a, bufferA->contents(), size * sizeof(T)); }

  if (!config.is_a_on_device) { bufferA->release(); }
  if (!config.is_b_on_device) { bufferB->release(); }

  return eIcicleError::SUCCESS;
}

/*********************************** ADD ***********************************/
constexpr char vector_add_kernel_name[] = "vector_add";
REGISTER_VECTOR_ADD_BACKEND("METAL", (metal_3vectors_op<scalar_t, vector_add_kernel_name>));

/*********************************** ACCUMULATE ***********************************/
constexpr char vector_accumulate_kernel_name[] = "vector_accumulate";
REGISTER_VECTOR_ACCUMULATE_BACKEND("METAL", (metal_2vectors_op<scalar_t, vector_accumulate_kernel_name>));

/*********************************** SUB ***********************************/
constexpr char vector_sub_kernel_name[] = "vector_sub";
REGISTER_VECTOR_SUB_BACKEND("METAL", (metal_3vectors_op<scalar_t, vector_sub_kernel_name>));

/*********************************** MUL ***********************************/
constexpr char vector_mul_kernel_name[] = "vector_mul";
REGISTER_VECTOR_MUL_BACKEND("METAL", (metal_3vectors_op<scalar_t, vector_mul_kernel_name>));

/*********************************** DIV ***********************************/
constexpr char vector_div_kernel_name[] = "vector_div";
REGISTER_VECTOR_DIV_BACKEND("METAL", (metal_3vectors_op<scalar_t, vector_div_kernel_name>));

constexpr char convert_montgomery_kernel_name[] = "convert_montgomery";
REGISTER_CONVERT_MONTGOMERY_BACKEND("METAL", (metal_convert_montgomery<scalar_t, convert_montgomery_kernel_name>));

/*********************************** SUM ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError
metal_vector_reduce(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  // size is vector size, not the total size
  uint64_t batch_size = config.batch_size;
  uint64_t vec_size = size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device
                              ? metal::map_to_metal_buffer(vec_a)
                              : metal::allocate_and_copy_to_device(vec_a, batch_size * vec_size * sizeof(T));
  auto [bufferRes, offsetRes] =
    config.is_result_on_device ? metal::map_to_metal_buffer(output) : metal::allocate_on_device(batch_size * sizeof(T));
  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferRes, offsetRes, 1);
  // Bind constants
  encoder->setBytes(&vec_size /*data*/, sizeof(vec_size) /*length*/, 2 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 3 /*index*/);
  const bool columns_batch = config.columns_batch;
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 4 /*index*/);

  MTL::Size threadsPerGroup(256, 1, 1);
  MTL::Size threadgroups(batch_size, 1, 1);

  // encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(output, bufferRes->contents(), batch_size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char vector_sum_kernel_name[] = "vector_sum";
REGISTER_VECTOR_SUM_BACKEND("METAL", (metal_vector_reduce<scalar_t, vector_sum_kernel_name>));

constexpr char vector_product_kernel_name[] = "vector_product";
REGISTER_VECTOR_PRODUCT_BACKEND("METAL", (metal_vector_reduce<scalar_t, vector_product_kernel_name>));

/*********************************** Generic Scalar Vector operation  ***********************************/

template <typename T, const char* KERNEL_NAME>
eIcicleError metal_scalar_op(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  uint64_t vec_size = size;
  uint64_t batch_size = config.batch_size;
  bool columns_batch = config.columns_batch;
  size *= config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device
                              ? metal::map_to_metal_buffer(scalar_a)
                              : metal::allocate_and_copy_to_device(scalar_a, config.batch_size * sizeof(T));
  auto [bufferB, offsetB] = config.is_b_on_device ? metal::map_to_metal_buffer(vec_b)
                                                  : metal::allocate_and_copy_to_device(vec_b, size * sizeof(T));
  auto [bufferRes, offsetRes] =
    config.is_result_on_device ? metal::map_to_metal_buffer(output) : metal::allocate_on_device(size * sizeof(T));

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferB, offsetB, 1);
  encoder->setBuffer(bufferRes, offsetRes, 2);
  // Bind constants
  encoder->setBytes(&vec_size /*data*/, sizeof(vec_size) /*length*/, 3 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 4 /*index*/);
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 5 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(output, bufferRes->contents(), size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }
  if (!config.is_b_on_device) { bufferB->release(); }

  return eIcicleError::SUCCESS;
}

/*********************************** Scalar + Vector ***********************************/
constexpr char scalar_add_kernel_name[] = "scalar_add";
REGISTER_SCALAR_ADD_VEC_BACKEND("METAL", (metal_scalar_op<scalar_t, scalar_add_kernel_name>));

/*********************************** Scalar - Vector ***********************************/
constexpr char scalar_sub_kernel_name[] = "scalar_sub";
REGISTER_SCALAR_SUB_VEC_BACKEND("METAL", (metal_scalar_op<scalar_t, scalar_sub_kernel_name>));

/*********************************** Scalar * Vector ***********************************/
constexpr char scalar_mul_kernel_name[] = "scalar_mul";
REGISTER_SCALAR_MUL_VEC_BACKEND("METAL", (metal_scalar_op<scalar_t, scalar_mul_kernel_name>));

/*********************************** TRANSPOSE ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError metal_matrix_transpose(
  const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  uint64_t batch_size = config.batch_size;
  bool columns_batch = config.columns_batch;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] =
    config.is_a_on_device ? metal::map_to_metal_buffer(mat_in)
                          : metal::allocate_and_copy_to_device(mat_in, nof_cols * nof_rows * batch_size * sizeof(T));
  auto [bufferRes, offsetRes] = config.is_result_on_device
                                  ? metal::map_to_metal_buffer(mat_out)
                                  : metal::allocate_on_device(nof_cols * nof_rows * batch_size * sizeof(T));

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferRes, offsetRes, 1);
  // Bind constants
  encoder->setBytes(&nof_rows /*data*/, sizeof(nof_rows) /*length*/, 2 /*index*/);
  encoder->setBytes(&nof_cols /*data*/, sizeof(nof_cols) /*length*/, 3 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 4 /*index*/);
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 5 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (nof_cols * nof_rows * batch_size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(mat_out, bufferRes->contents(), nof_cols * nof_rows * batch_size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char matrix_transpose_kernel_name[] = "matrix_transpose";
REGISTER_MATRIX_TRANSPOSE_BACKEND("METAL", (metal_matrix_transpose<scalar_t, matrix_transpose_kernel_name>));

/*********************************** BIT REVERSE ***********************************/
uint64_t count_leading_zeros(uint64_t x)
{
  // Count leading zeros using a binary search approach
  if (x == 0) return 64; // Special case for zero

  uint64_t n = 0;
  if (x <= 0x00000000FFFFFFFF) {
    n += 32;
    x <<= 32;
  }
  if (x <= 0x0000FFFFFFFFFFFF) {
    n += 16;
    x <<= 16;
  }
  if (x <= 0x00FFFFFFFFFFFFFF) {
    n += 8;
    x <<= 8;
  }
  if (x <= 0x0FFFFFFFFFFFFFFF) {
    n += 4;
    x <<= 4;
  }
  if (x <= 0x3FFFFFFFFFFFFFFF) {
    n += 2;
    x <<= 2;
  }
  if (x <= 0x7FFFFFFFFFFFFFFF) {
    n += 1;
    x <<= 1;
  }
  return n;
}

template <typename T, const char* KERNEL_NAME_INPLACE, const char* KERNEL_NAME_OUTPLACE>
eIcicleError
metal_bit_reverse(const Device& device, const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out)
{
  uint64_t batch_size = config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }
  const bool is_in_place = (vec_in == vec_out);

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device
                              ? metal::map_to_metal_buffer(vec_in)
                              : metal::allocate_and_copy_to_device(vec_in, batch_size * size * sizeof(T));
  MTL::Buffer* bufferRes = nullptr;
  size_t offsetRes = 0;
  if (!is_in_place) {
    std::tie(bufferRes, offsetRes) = config.is_result_on_device
                                       ? metal::map_to_metal_buffer(vec_out)
                                       : metal::allocate_on_device(batch_size * size * sizeof(T));
  }

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object that
  // we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(is_in_place ? KERNEL_NAME_INPLACE : KERNEL_NAME_OUTPLACE);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);

  uint64_t shift = count_leading_zeros(size) + 1;
  const bool columns_batch = config.columns_batch;
  if (is_in_place) {
    encoder->setBuffer(bufferA, offsetA, 0);
    encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 1 /*index*/);
    encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 2 /*index*/);
    encoder->setBytes(&shift /*data*/, sizeof(shift) /*length*/, 3 /*index*/);
    encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 4 /*index*/);
  } else {
    encoder->setBuffer(bufferA, offsetA, 0);
    encoder->setBuffer(bufferRes, offsetRes, 1);
    encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 2 /*index*/);
    encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 3 /*index*/);
    encoder->setBytes(&shift /*data*/, sizeof(shift) /*length*/, 4 /*index*/);
    encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 5 /*index*/);
  }

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (batch_size * size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    if (is_in_place) {
      std::memcpy(vec_out, bufferA->contents(), batch_size * size * sizeof(T));
    } else {
      std::memcpy(vec_out, bufferRes->contents(), batch_size * size * sizeof(T));
    }
  }

  if (!config.is_a_on_device) { bufferA->release(); }
  if (!config.is_result_on_device && !is_in_place) { bufferRes->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char bit_reverse_kernel_name_inplace[] = "bit_reverse_in_place";
constexpr char bit_reverse_kernel_name_outplace[] = "bit_reverse";
REGISTER_BIT_REVERSE_BACKEND(
  "METAL", (metal_bit_reverse<scalar_t, bit_reverse_kernel_name_inplace, bit_reverse_kernel_name_outplace>));

/*********************************** SLICE ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError metal_slice(
  const Device& device,
  const T* vec_a,
  uint64_t offset,
  uint64_t stride,
  uint64_t size_in,
  uint64_t size_out,
  const VecOpsConfig& config,
  T* vec_out)
{
  uint64_t batch_size = config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device
                              ? metal::map_to_metal_buffer(vec_a)
                              : metal::allocate_and_copy_to_device(vec_a, size_in * batch_size * sizeof(T));

  auto [bufferRes, offsetRes] = config.is_result_on_device
                                  ? metal::map_to_metal_buffer(vec_out)
                                  : metal::allocate_on_device(size_out * batch_size * sizeof(T));

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object
  // that we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferRes, offsetRes, 1);
  // Bind constants
  encoder->setBytes(&offset /*data*/, sizeof(offset) /*length*/, 2 /*index*/);
  encoder->setBytes(&stride /*data*/, sizeof(stride) /*length*/, 3 /*index*/);
  encoder->setBytes(&size_in /*data*/, sizeof(size_in) /*length*/, 4 /*index*/);
  encoder->setBytes(&size_out /*data*/, sizeof(size_out) /*length*/, 5 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 6 /*index*/);
  const bool columns_batch = config.columns_batch;
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 7 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size_in * batch_size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(vec_out, bufferRes->contents(), size_out * batch_size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char slice_kernel_name[] = "slice";
REGISTER_SLICE_BACKEND("METAL", (metal_slice<scalar_t, slice_kernel_name>));

/*********************************** Highest non-zero idx ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError metal_highest_non_zero_idx(
  const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
{
  // size is vector size, not the total size
  uint64_t batch_size = config.batch_size;
  uint64_t vec_size = size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device
                              ? metal::map_to_metal_buffer(vec_a)
                              : metal::allocate_and_copy_to_device(vec_a, batch_size * vec_size * sizeof(T));
  auto [bufferRes, offsetRes] = config.is_result_on_device ? metal::map_to_metal_buffer(out_idx)
                                                           : metal::allocate_on_device(batch_size * sizeof(int64_t));
  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object
  // that we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferRes, offsetRes, 1);
  // Bind constants
  encoder->setBytes(&vec_size /*data*/, sizeof(vec_size) /*length*/, 2 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 3 /*index*/);
  const bool columns_batch = config.columns_batch;
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 4 /*index*/);

  MTL::Size threadsPerGroup(256, 1, 1); // must be a power of 2
  MTL::Size threadgroups(batch_size, 1, 1);

  // encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(out_idx, bufferRes->contents(), batch_size * sizeof(int64_t));
    bufferRes->release();
    ICICLE_LOG_DEBUG << "Highest non-zero idx: " << out_idx[0];
  }

  if (!config.is_a_on_device) { bufferA->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char highest_non_zero_idx_kernel_name[] = "highest_non_zero_idx";
REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND(
  "METAL", (metal_highest_non_zero_idx<scalar_t, highest_non_zero_idx_kernel_name>));

/*********************************** Polynomial evaluation ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError metal_poly_eval(
  const Device& device,
  const T* coeffs,
  uint64_t coeffs_size,
  const T* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  T* evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  uint64_t batch_size = config.batch_size;
  uint64_t size = coeffs_size * batch_size;
  bool columns_batch = config.columns_batch;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  // interpret config's a,b as in CUDA
  uint64_t total_result_size = domain_size * batch_size;

  auto [bufferCoeffs, offsetCoeffs] = config.is_a_on_device
                                        ? metal::map_to_metal_buffer(coeffs)
                                        : metal::allocate_and_copy_to_device(coeffs, size * sizeof(T));
  auto [bufferDomain, offsetDomain] = config.is_b_on_device
                                        ? metal::map_to_metal_buffer(domain)
                                        : metal::allocate_and_copy_to_device(domain, domain_size * sizeof(T));
  auto [bufferEvals, offsetEvals] = config.is_result_on_device
                                      ? metal::map_to_metal_buffer(evals)
                                      : metal::allocate_on_device(total_result_size * sizeof(T));

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object
  // that we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferCoeffs, offsetCoeffs, 0);
  encoder->setBuffer(bufferDomain, offsetDomain, 1);
  encoder->setBuffer(bufferEvals, offsetEvals, 2);
  // Bind constants
  encoder->setBytes(&coeffs_size /*data*/, sizeof(coeffs_size) /*length*/, 3 /*index*/);
  encoder->setBytes(&batch_size /*data*/, sizeof(batch_size) /*length*/, 4 /*index*/);
  encoder->setBytes(&columns_batch /*data*/, sizeof(columns_batch) /*length*/, 5 /*index*/);
  encoder->setBytes(&domain_size /*data*/, sizeof(domain_size) /*length*/, 6 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (total_result_size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(evals, bufferEvals->contents(), total_result_size * sizeof(T));
    bufferEvals->release();
  }

  if (!config.is_a_on_device) { bufferCoeffs->release(); }
  if (!config.is_b_on_device) { bufferDomain->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char poly_eval_kernel_name[] = "poly_eval";
REGISTER_POLYNOMIAL_EVAL("METAL", (metal_poly_eval<scalar_t, poly_eval_kernel_name>));

/*********************************** Polynomial division ***********************************/
template <typename T, const char* KERNEL_NAME>
eIcicleError metal_poly_divide(
  const Device& device,
  const T* numerator,
  uint64_t numerator_size,
  const T* denominator,
  uint64_t denominator_size,
  const VecOpsConfig& config,
  T* q_out,
  uint64_t q_size,
  T* r_out,
  uint64_t r_size)
{
  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_VERBOSE << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }
  if (config.batch_size != 1) {
    ICICLE_LOG_ERROR << "polynomial division is not implemented for batch. Planned for TBD";
    return eIcicleError::API_NOT_IMPLEMENTED;
  }
  if (config.batch_size != 1 && config.columns_batch) {
    ICICLE_LOG_ERROR << "polynomial division is not implemented for column batch. Planned for TBD";
    return eIcicleError::API_NOT_IMPLEMENTED;
  }

  // compute degrees of numerator and denumerator
  int64_t numerator_deg, denominator_deg;
  auto degree_config = default_vec_ops_config();
  degree_config.stream = config.stream;
  degree_config.is_async = config.is_async;
  degree_config.is_a_on_device = config.is_a_on_device; // numerator
  metal_highest_non_zero_idx<T, highest_non_zero_idx_kernel_name>(
    device, numerator, numerator_size, degree_config, &numerator_deg);
  ICICLE_LOG_DEBUG << "GPU: numerator_deg: " << numerator_deg;
  degree_config.is_a_on_device = config.is_b_on_device; // denominator
  metal_highest_non_zero_idx<T, highest_non_zero_idx_kernel_name>(
    device, denominator, denominator_size, degree_config, &denominator_deg);
  ICICLE_LOG_DEBUG << "GPU: denominator_deg: " << denominator_deg;

  // verify outputs are large enough
  ICICLE_ASSERT(r_size >= (1 + denominator_deg))
    << "polynomial division expects r(x) size to be similar to numerator(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denominator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denominator)+1";

  // Create Metal buffers
  auto [bufferDenominator, offsetDenominator] =
    config.is_b_on_device ? metal::map_to_metal_buffer(denominator)
                          : metal::allocate_and_copy_to_device(denominator, denominator_size * sizeof(T));

  auto [bufferQ, offsetQ] =
    config.is_result_on_device ? metal::map_to_metal_buffer(q_out) : metal::allocate_on_device(q_size * sizeof(T));

  if (config.is_result_on_device) {
    // initialize r_out with numerator
    for (int i = 0; i < 1 + numerator_deg; i++) {
      r_out[i] = numerator[i];
    }
  }

  auto [bufferR, offsetR] =
    config.is_result_on_device ? metal::map_to_metal_buffer(r_out) : metal::allocate_on_device(r_size * sizeof(T));

  if (!config.is_result_on_device) {
    // initialize bufferR with numerator.
    std::memcpy((void*)bufferR->contents(), numerator, (1 + numerator_deg) * sizeof(T));
  }

  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  // METAL_GET_PIPELINE() macro automatically loads the correct library and returns a Metal compute-pipeline object
  // that we need to dispatch the kernel
  auto pipelineState = METAL_GET_PIPELINE(KERNEL_NAME);

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);

  // Bind buffers
  encoder->setBuffer(bufferR, offsetR, 0);
  encoder->setBuffer(bufferQ, offsetQ, 1);
  encoder->setBuffer(bufferDenominator, offsetDenominator, 2);

  // Bind constants
  int64_t deg_r = numerator_deg;
  encoder->setBytes(&deg_r /*data*/, sizeof(deg_r) /*length*/, 3 /*index*/);
  encoder->setBytes(&denominator_deg /*data*/, sizeof(denominator_deg) /*length*/, 4 /*index*/);

  // NOTE (forwarded from CUDA, TODO: is this relevant to CUDA?): this kernel works only with a single block due to
  // intra-block synchronization school_book_division<<<1, 32 /* Empirical best value*/, 0, cuda_stream>>>(d_r_out,
  // d_q_out, denominator, deg_r, denominator_deg);
  const int NOF_THREADS_PER_BLOCK = 32;
  const int NOF_BLOCKS = 1; //(total_result_size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    // Seems that this is a double-free so not required
    // commandBuffer->addCompletedHandler([](MTL::CommandBuffer* commandBuffer) {
    //   commandBuffer->release();
    // });
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(r_out, bufferR->contents(), r_size * sizeof(T));
    bufferR->release();
    std::memcpy(q_out, bufferQ->contents(), q_size * sizeof(T));
    bufferQ->release();
  }

  if (!config.is_b_on_device) { bufferDenominator->release(); }

  return eIcicleError::SUCCESS;
}

constexpr char poly_divide_kernel_name[] = "poly_divide";
REGISTER_POLYNOMIAL_DIVISION("METAL", (metal_poly_divide<scalar_t, poly_divide_kernel_name>));

#ifdef EXT_FIELD
constexpr char extension_vector_add_kernel_name[] = "extension_vector_add";
constexpr char extension_vector_accumulate_kernel_name[] = "extension_vector_accumulate";
constexpr char extension_vector_sub_kernel_name[] = "extension_vector_sub";
constexpr char extension_vector_mul_kernel_name[] = "extension_vector_mul";
constexpr char extension_vector_div_kernel_name[] = "extension_vector_div";
constexpr char extension_convert_montgomery_kernel_name[] = "extension_convert_montgomery";
constexpr char extension_vector_sum_kernel_name[] = "extension_vector_sum";
constexpr char extension_vector_product_kernel_name[] = "extension_vector_product";
constexpr char extension_scalar_mul_kernel_name[] = "extension_scalar_mul";
constexpr char extension_scalar_add_kernel_name[] = "extension_scalar_add";
constexpr char extension_scalar_sub_kernel_name[] = "extension_scalar_sub";
constexpr char extension_matrix_transpose_kernel_name[] = "extension_matrix_transpose";
constexpr char extension_bit_reverse_kernel_name_inplace[] = "extension_bit_reverse_in_place";
constexpr char extension_bit_reverse_kernel_name_outplace[] = "extension_bit_reverse";
constexpr char extension_slice_kernel_name[] = "extension_slice";

REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("METAL", (metal_3vectors_op<extension_t, extension_vector_add_kernel_name>));
REGISTER_VECTOR_ACCUMULATE_EXT_FIELD_BACKEND(
  "METAL", (metal_2vectors_op<extension_t, extension_vector_accumulate_kernel_name>));
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("METAL", (metal_3vectors_op<extension_t, extension_vector_sub_kernel_name>));
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("METAL", (metal_3vectors_op<extension_t, extension_vector_mul_kernel_name>));
REGISTER_VECTOR_DIV_EXT_FIELD_BACKEND("METAL", (metal_3vectors_op<extension_t, extension_vector_div_kernel_name>));
REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND(
  "METAL", (metal_convert_montgomery<extension_t, extension_convert_montgomery_kernel_name>));
REGISTER_VECTOR_SUM_EXT_FIELD_BACKEND("METAL", (metal_vector_reduce<extension_t, extension_vector_sum_kernel_name>));
REGISTER_VECTOR_PRODUCT_EXT_FIELD_BACKEND(
  "METAL", (metal_vector_reduce<extension_t, extension_vector_product_kernel_name>));
REGISTER_SCALAR_ADD_VEC_EXT_FIELD_BACKEND("METAL", (metal_scalar_op<extension_t, extension_scalar_add_kernel_name>));
REGISTER_SCALAR_SUB_VEC_EXT_FIELD_BACKEND("METAL", (metal_scalar_op<extension_t, extension_scalar_sub_kernel_name>));
REGISTER_SCALAR_MUL_VEC_EXT_FIELD_BACKEND("METAL", (metal_scalar_op<extension_t, extension_scalar_mul_kernel_name>));
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND(
  "METAL", (metal_matrix_transpose<extension_t, extension_matrix_transpose_kernel_name>));
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND(
  "METAL",
  (metal_bit_reverse<
    extension_t,
    extension_bit_reverse_kernel_name_inplace,
    extension_bit_reverse_kernel_name_outplace>));
REGISTER_SLICE_EXT_FIELD_BACKEND("METAL", (metal_slice<extension_t, extension_slice_kernel_name>));
#endif // EXT_FIELD