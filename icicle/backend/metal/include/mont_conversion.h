#pragma once

#include "Metal/Metal.h"

#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "metal_device_api.h"
#include "metal_library_loader.h"

template <typename T, const char* KERNEL_NAME>
eIcicleError metal_convert_montgomery(
  const Device& device, const T* vec_a, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output)
{
  size *= config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_result_on_device) {
      ICICLE_LOG_WARNING << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device ? metal::map_to_metal_buffer(vec_a)
                                                  : metal::allocate_and_copy_to_device(vec_a, size * sizeof(T));
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
  encoder->setBuffer(bufferRes, offsetRes, 1);
  // Bind constants
  encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 2 /*index*/);
  encoder->setBytes(&is_to_montgomery /*data*/, sizeof(is_to_montgomery) /*length*/, 3 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    commandBuffer->commit();
    return eIcicleError::SUCCESS;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(output, bufferRes->contents(), size * sizeof(T));
    bufferRes->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }

  return eIcicleError::SUCCESS;
}