#pragma once

#include "Metal/Metal.h"

#include "icicle/vec_ops.h"
#include "icicle/backend/msm_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "metal_device_api.h"
#include "metal_library_loader.h"
#include "metal_msm.h"

template <typename S, typename P, typename A, const bool IsG2 = false>
class MetalMSMBackend
{
private:
  const unsigned m_single_msm_size;
  const MSMConfig& m_config;

  MTL::CommandQueue* const m_cmdQueue = (m_config.stream != nullptr)
                                          ? reinterpret_cast<MTL::CommandQueue* const>(m_config.stream)
                                          : metal::get_default_command_queue();
  // Derived constants
  uint32_t m_nof_scalars; // todo - add c and command queue
  uint32_t m_nof_points;
  uint32_t m_total_bms_per_msm;
  uint32_t m_nof_bms_per_msm;
  uint32_t m_input_indexes_count;
  uint32_t m_bm_bitsize;
  uint32_t m_nof_buckets;
  uint32_t m_total_nof_buckets;
  uint32_t m_nof_bms_in_batch;
  uint32_t m_c;
  uint32_t m_bitsize;
  uint32_t m_nof_buckets_to_compute;
  uint32_t m_zero_bucket_offset;

  // Metal buffers
  MTL::Buffer* m_scalars_buffer = nullptr;
  size_t m_scalars_offset = 0;
  MTL::Buffer* m_points_buffer = nullptr;
  size_t m_points_offset = 0;
  MTL::Buffer* m_results_buffer = nullptr;
  size_t m_results_offset = 0;

  MTL::Buffer* m_bucket_offsets_buffer = nullptr;
  MTL::Buffer* m_single_bucket_indices_buffer = nullptr;
  MTL::Buffer* m_bucket_sizes_buffer = nullptr;
  // MTL::Buffer* m_nof_buckets_to_compute_buffer = nullptr;
  MTL::Buffer* m_buckets_buffer = nullptr;
  MTL::Buffer* m_bucket_indices_buffer = nullptr;
  MTL::Buffer* m_point_indices_buffer = nullptr;
  MTL::Buffer* m_final_result = nullptr;
  size_t m_dummy_offset = 0;

public:
  MetalMSMBackend(int single_msm_size, const MSMConfig& config) : m_single_msm_size(single_msm_size), m_config(config)
  {
    m_c = (config.c == 0) ? get_optimal_c(single_msm_size) : config.c;
    m_bitsize = (config.bitsize == 0) ? S::NBITS : config.bitsize;
    m_nof_scalars = m_config.batch_size * m_single_msm_size; // assuming scalars not shared between batch elements
    // TODO: introduce precompute
    // m_nof_points = (m_config.are_points_shared_in_batch ? single_msm_size : m_nof_scalars) *
    // m_config.precompute_factor;
    m_nof_points = m_config.are_points_shared_in_batch ? single_msm_size : m_nof_scalars;
    m_total_bms_per_msm = (m_bitsize + m_c - 1) / m_c;
    // TODO: introduce precompute
    // m_nof_bms_per_msm = (m_total_bms_per_msm - 1) / m_config.precompute_factor + 1;
    m_nof_bms_per_msm = m_total_bms_per_msm;
    m_input_indexes_count = m_nof_scalars * m_total_bms_per_msm;
    m_bm_bitsize = (uint32_t)ceil(std::log2(m_nof_bms_per_msm));
    m_nof_buckets =
      (m_nof_bms_per_msm << m_c) -
      m_nof_bms_per_msm; // minus nof_bms_per_msm because zero bucket is not included in each bucket module
    m_total_nof_buckets = m_nof_buckets * m_config.batch_size;
    m_nof_bms_in_batch = m_nof_bms_per_msm * m_config.batch_size;

    ICICLE_LOG_DEBUG << "m_nof_scalars: " << m_nof_scalars;
    ICICLE_LOG_DEBUG << "m_nof_points: " << m_nof_points;
    ICICLE_LOG_DEBUG << "m_total_bms_per_msm: " << m_total_bms_per_msm;
    ICICLE_LOG_DEBUG << "m_nof_bms_per_msm: " << m_nof_bms_per_msm;
    ICICLE_LOG_DEBUG << "m_input_indexes_count: " << m_input_indexes_count;
    ICICLE_LOG_DEBUG << "m_bm_bitsize: " << m_bm_bitsize;
    ICICLE_LOG_DEBUG << "m_nof_buckets: " << m_nof_buckets;
    ICICLE_LOG_DEBUG << "m_total_nof_buckets: " << m_total_nof_buckets;
    ICICLE_LOG_DEBUG << "m_nof_bms_in_batch: " << m_nof_bms_in_batch;
    ICICLE_LOG_DEBUG << "m_c: " << m_c;
    ICICLE_LOG_DEBUG << "m_bitsize: " << m_bitsize;
  }

  ~MetalMSMBackend()
  {
    if (!m_config.are_scalars_on_device) { m_scalars_buffer->release(); }
    if (!m_config.are_points_on_device) { m_points_buffer->release(); }
  }

  // TODO find optimal values
  static uint32_t get_optimal_c(int msm_size)
  {
    // return (uint32_t)std::min(std::max(std::ceil(std::log2(msm_size)) - 4.0, 4.0), 20.0);
    // before implementing the large buckets optimization we use this list for optimal c for bn254 (these are c values
    // that divide 254 with a large reminder):
    int log_size = std::ceil(std::log2(msm_size));
    if (log_size < 15) { return 8; }
    if (log_size < 17) { return 13; }
    if (log_size < 20) { return 15; }
    if (log_size < 22) { return 16; }
    if (log_size < 24) { return 17; }
    return 20;
  } // 20 as the largest value is a temporary fix for the large buckets problem

  void from_montgomery_on_device(MTL::Buffer* d_input, int size, MTL::Buffer* d_output, bool affine) const
  {
    const char* kernel_name = IsG2 ? (affine ? "affine_convert_montgomery_g2" : "convert_montgomery_g2")
                                   : (affine ? "affine_convert_montgomery" : "convert_montgomery");
    auto pipelineState = METAL_GET_PIPELINE(kernel_name);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    // Bind buffers
    encoder->setBuffer(d_input, 0, 0);
    encoder->setBuffer(d_output, 0, 1);
    // Bind constants
    uint64_t big_size = size;
    encoder->setBytes(&big_size /*data*/, sizeof(uint64_t) /*length*/, 2 /*index*/);
    bool is_to_montgomery = false;
    encoder->setBytes(&is_to_montgomery /*data*/, sizeof(is_to_montgomery) /*length*/, 3 /*index*/);

    const int NOF_THREADS_PER_BLOCK = 256;
    const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
    MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

  eIcicleError upload_scalars(const S* scalars)
  {
    std::tie(m_scalars_buffer, m_scalars_offset) =
      m_config.are_scalars_on_device
        ? metal::map_to_metal_buffer(scalars)
        : metal::allocate_and_copy_to_device(
            scalars, m_nof_scalars * sizeof(S)); // TODO - would it be better to use private buffer?
    if (m_config.are_scalars_montgomery_form) {
      from_montgomery_on_device(m_scalars_buffer, m_nof_scalars, m_scalars_buffer, false /* affine */);
    }
    return eIcicleError::SUCCESS;
  }

  eIcicleError upload_points(const A* points)
  {
    std::tie(m_points_buffer, m_points_offset) =
      m_config.are_points_on_device
        ? metal::map_to_metal_buffer(points)
        : metal::allocate_and_copy_to_device(
            points, m_nof_points * sizeof(A)); // TODO - would it be better to use private buffer?
    if (m_config.are_points_montgomery_form) {
      from_montgomery_on_device(m_points_buffer, m_nof_points, m_points_buffer, true /* affine */);
    }
    return eIcicleError::SUCCESS;
  }

  // Perform counting sort on a specific 4-bit digit (hexadecimal place)
  void countingSort(uint32_t* arr, int shift, size_t numElements, uint32_t* twin = nullptr, uint32_t* triplet = nullptr)
  {
    const int base = 16;
    uint32_t* output = new uint32_t[numElements]; // Temporary output array
    uint32_t* twin_output = twin ? new uint32_t[numElements] : nullptr;
    uint32_t* triplet_output = triplet ? new uint32_t[numElements] : nullptr;
    int count[base] = {0};

    // Count occurrences of each hex digit
    for (size_t i = 0; i < numElements; i++) {
      int digit = (arr[i] >> shift) & 0xF;
      count[digit]++;
    }

    // Convert count array to prefix sum for stable sorting
    for (int i = 1; i < base; i++) {
      count[i] += count[i - 1];
    }

    // Build the sorted array
    for (int i = numElements - 1; i >= 0; i--) {
      int digit = (arr[i] >> shift) & 0xF;
      output[--count[digit]] = arr[i];
      if (twin) twin_output[count[digit]] = twin[i];
      if (triplet) triplet_output[count[digit]] = triplet[i];
    }

    // Copy back to input array
    std::memcpy(arr, output, numElements * sizeof(uint32_t));
    if (twin) std::memcpy(twin, twin_output, numElements * sizeof(uint32_t));
    if (triplet) std::memcpy(triplet, triplet_output, numElements * sizeof(uint32_t));

    // Free temporary arrays
    delete[] output;
    delete[] twin_output;
    delete[] triplet_output;
  }

  // Radix sort for base 16 using bitwise operations
  void radixSort(uint32_t* arr, size_t numElements, uint32_t* twin = nullptr, uint32_t* triplet = nullptr)
  {
    if (numElements == 0) return;

    for (int shift = 0; shift < 32; shift += 4) {
      // int shift = 0;
      countingSort(arr, shift, numElements, twin, triplet);
    }
  }

  void sortWithMetalSharedBuffer(
    MTL::Buffer* bufferArr,
    MTL::Buffer* bufferTwin,
    MTL::Buffer* bufferTriplet,
    size_t numElements,
    uint32_t offset = 0)
  {
    // Get raw pointers directly from Metal buffers
    uint32_t* arr = static_cast<uint32_t*>(bufferArr->contents()) + offset;
    uint32_t* twin = bufferTwin ? static_cast<uint32_t*>(bufferTwin->contents()) + offset : nullptr;
    uint32_t* triplet = bufferTriplet ? static_cast<uint32_t*>(bufferTriplet->contents()) + offset : nullptr;

    // Directly modify the buffer contents
    radixSort(arr, numElements, twin, triplet);
  }

  // Function to find the next power of 2 greater than or equal to n
  uint32_t nextPowerOf2(uint32_t n)
  {
    if (n == 0) return 1;
    uint32_t p = 1;
    while (p < n)
      p <<= 1;
    return p;
  }

  // Metal-based Exclusive Scan
  void exclusiveScan2(MTL::Buffer* input_buffer, MTL::Buffer* output_buffer, uint32_t size, bool inclusive = false)
  {
    MTL::ComputePipelineState* upsweepPipeline = METAL_GET_PIPELINE("device_exclusive_scan_upsweep")
      MTL::ComputePipelineState* downsweepPipeline = METAL_GET_PIPELINE("device_exclusive_scan_downsweep")

        uint32_t* input = static_cast<uint32_t*>(input_buffer->contents());
    uint32_t* output = static_cast<uint32_t*>(output_buffer->contents());

    uint32_t paddedSize = nextPowerOf2(size);
    uint32_t inputSize = paddedSize + (inclusive ? 1 : 0);
    uint32_t numRounds = log2(paddedSize);

    // Allocate memory for input and output
    auto [temp_buffer, temp_offset] = metal::allocate_on_device(inputSize * sizeof(uint32_t));
    std::memcpy(temp_buffer->contents(), input, size * sizeof(uint32_t));

    uint32_t num_threads;
    uint32_t num_threads_per_group = 256; // TODO - find optimal number

    // **UPSWEEP PHASE**
    for (int stride = 1; stride < paddedSize; stride <<= 1) {
      bool last = stride == paddedSize / 2;
      MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();
      MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();

      encoder->setComputePipelineState(upsweepPipeline);
      encoder->setBuffer(temp_buffer, temp_offset, 0);
      encoder->setBytes(&paddedSize, sizeof(uint32_t), 1);
      encoder->setBytes(&stride, sizeof(uint32_t), 2);
      encoder->setBytes(&last, sizeof(bool), 3);
      encoder->setBytes(&inclusive, sizeof(bool), 4);
      num_threads = paddedSize / (2 * stride);
      int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
      MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
      MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);
      encoder->dispatchThreadgroups(gridSize, threadGroupSize);
      encoder->endEncoding();
      cmdBuffer->commit();
      cmdBuffer->waitUntilCompleted();
    }

    // **DOWNSWEEP PHASE**
    for (int stride = paddedSize / 2; stride > 0; stride >>= 1) {
      // uint32_t stride = paddedSize/2;
      MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();
      MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();

      encoder->setComputePipelineState(downsweepPipeline);
      encoder->setBuffer(temp_buffer, temp_offset, 0);
      encoder->setBytes(&paddedSize, sizeof(uint32_t), 1);
      encoder->setBytes(&stride, sizeof(uint32_t), 2);

      num_threads = paddedSize / (2 * stride);
      int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
      MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
      MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);
      encoder->dispatchThreadgroups(gridSize, threadGroupSize);
      encoder->endEncoding();
      cmdBuffer->commit();
      cmdBuffer->waitUntilCompleted();
    }

    // Copy result back to output array
    if (inclusive) {
      memcpy(output, static_cast<uint32_t*>(temp_buffer->contents()) + 1, size * sizeof(uint32_t));
    } else {
      memcpy(output, temp_buffer->contents(), size * sizeof(uint32_t));
    }

    // Cleanup
    temp_buffer->release();
  }

  void split_scalars()
  {
    const uint32_t num_threads = m_nof_scalars;
    const uint32_t num_threads_per_group = 256; // TODO - find optimal number

    auto pipelineState = METAL_GET_PIPELINE(IsG2 ? "split_scalars_kernel_g2" : "split_scalars_kernel");

    int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    SplitScalarsParams params = {
      m_nof_scalars,
      m_nof_points,
      m_single_msm_size,
      m_total_bms_per_msm,
      m_bm_bitsize,
      m_c,
      // TODO: precompute
      //   (unsigned)m_config.precompute_factor,
      1,
      m_nof_bms_per_msm,
    };

    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(m_bucket_indices_buffer, 0, 0);
    encoder->setBuffer(m_point_indices_buffer, 0, 1);
    encoder->setBuffer(m_scalars_buffer, m_scalars_offset, 2);
    encoder->setBytes(&params, sizeof(SplitScalarsParams), 3);

    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

  void backward_mask(MTL::Buffer* back_mask_buffer)
  {
    // Load default library and get the kernel function
    auto pipelineState3 = METAL_GET_PIPELINE("backward_mask_kernel");

    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState3);

    // Set buffers
    encoder->setBuffer(m_bucket_indices_buffer, 0, 0);
    encoder->setBuffer(back_mask_buffer, 0, 1);

    // Dispatch threads: use one threadgroup of 16 threads for 16 elements.
    const uint32_t num_threads = m_input_indexes_count;
    const uint32_t num_threads_per_group = 256; // TODO - find optimal number

    const uint32_t nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    auto gridSize = MTL::Size(nof_blocks, 1, 1);
    auto threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

  void compact_kernel(MTL::Buffer* totalRunsBuffer, MTL::Buffer* scannedMaskBuffer)
  {
    // Load default library and get the kernel function
    auto pipelineStateComp = METAL_GET_PIPELINE("compact_kernel");

    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineStateComp);

    // Set buffers
    encoder->setBuffer(scannedMaskBuffer, 0, 0);
    encoder->setBuffer(m_bucket_offsets_buffer, 0, 1);
    encoder->setBuffer(totalRunsBuffer, 0, 2);
    encoder->setBytes(&m_input_indexes_count, sizeof(uint32_t), 3);

    // Dispatch threads: use one threadgroup of 16 threads for 16 elements.
    const uint32_t num_threads = m_input_indexes_count;
    const uint32_t num_threads_per_group = 256; // TODO - find optimal number
    const uint32_t nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    auto gridSize = MTL::Size(nof_blocks, 1, 1);
    auto threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

  void write_rle_output(MTL::Buffer* totalRunsBuffer)
  {
    auto pipelineState4 = METAL_GET_PIPELINE("write_rle_output");
    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineState4);

    // Set buffers
    encoder->setBuffer(m_bucket_indices_buffer, m_dummy_offset, 0);
    encoder->setBuffer(m_bucket_offsets_buffer, m_dummy_offset, 1);
    encoder->setBuffer(m_single_bucket_indices_buffer, m_dummy_offset, 2);
    encoder->setBuffer(m_bucket_sizes_buffer, m_dummy_offset, 3);
    encoder->setBuffer(totalRunsBuffer, m_dummy_offset, 4);

    // Dispatch threads: use one threadgroup of 16 threads for 16 elements.
    const uint32_t num_threads = m_input_indexes_count;
    const uint32_t num_threads_per_group = 256; // TODO - find optimal number
    const uint32_t nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    auto gridSize = MTL::Size(nof_blocks, 1, 1);
    auto threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

  eIcicleError split_and_sort_scalars()
  {
    std::tie(m_bucket_offsets_buffer, m_dummy_offset) =
      metal::allocate_on_device(sizeof(uint32_t) * (m_total_nof_buckets + 1));
    std::tie(m_single_bucket_indices_buffer, m_dummy_offset) =
      metal::allocate_on_device(sizeof(uint32_t) * (m_total_nof_buckets + 1));
    std::tie(m_bucket_sizes_buffer, m_dummy_offset) =
      metal::allocate_on_device(sizeof(uint32_t) * (m_total_nof_buckets + 1));
    // std::tie(m_nof_buckets_to_compute_buffer, m_dummy_offset) = metal::allocate_on_device(sizeof(uint32_t));
    std::tie(m_bucket_indices_buffer, m_dummy_offset) =
      metal::allocate_on_device(sizeof(uint32_t) * m_input_indexes_count);
    std::tie(m_point_indices_buffer, m_dummy_offset) =
      metal::allocate_on_device(sizeof(uint32_t) * m_input_indexes_count);

    split_scalars();

    m_nof_buckets_to_compute -= m_zero_bucket_offset;

    std::vector<std::thread> threads; // Vector to hold the threads

    // Sorting each backt module with a different thread. Note - does not support batch yet.
    for (int i = 0; i < m_total_bms_per_msm; i++) {
      threads.push_back(std::thread([i, this]() {
        sortWithMetalSharedBuffer(
          m_bucket_indices_buffer, m_point_indices_buffer, nullptr, m_nof_scalars, i * m_nof_scalars);
      }));
    }

    // Join all threads
    for (auto& t : threads) {
      t.join();
    }

    auto [back_mask_buffer, back_mask_offset] = metal::allocate_on_device(sizeof(uint32_t) * m_input_indexes_count);
    backward_mask(back_mask_buffer);

    auto [scannedMaskBuffer, scannedMaskOffset] = metal::allocate_on_device(m_input_indexes_count * sizeof(uint32_t));
    exclusiveScan2(back_mask_buffer, scannedMaskBuffer, m_input_indexes_count, true);

    auto [totalRunsBuffer, totalRunsOffset] = metal::allocate_on_device(sizeof(uint32_t));
    compact_kernel(totalRunsBuffer, scannedMaskBuffer);

    m_nof_buckets_to_compute = static_cast<uint32_t*>(totalRunsBuffer->contents())[0];
    write_rle_output(totalRunsBuffer);

    // Clean up
    scannedMaskBuffer->release();
    totalRunsBuffer->release();
    back_mask_buffer->release();
    return eIcicleError::SUCCESS;
  }

  eIcicleError buckets_init_and_sort()
  {
    std::tie(m_buckets_buffer, m_dummy_offset) =
      metal::allocate_on_device_private(sizeof(P) * (m_total_nof_buckets + m_nof_bms_in_batch));
    initialize_projective_buffer(m_buckets_buffer, m_total_nof_buckets + m_nof_bms_in_batch);

    // removing zero bucket, if it exists
    unsigned smallest_bucket_index = ((unsigned*)m_single_bucket_indices_buffer->contents())[0];
    // maybe zero bucket is empty after all? in this case zero_bucket_offset is set to 0
    m_zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0;

    m_nof_buckets_to_compute -= m_zero_bucket_offset;
    sortWithMetalSharedBuffer(
      m_bucket_sizes_buffer, m_bucket_offsets_buffer, m_single_bucket_indices_buffer, m_nof_buckets_to_compute,
      m_zero_bucket_offset);
    return eIcicleError::SUCCESS;
  }

  eIcicleError bucket_accumulation()
  {
    // ------------------------- Accumulation of (non-large) buckets ---------------------------------

    const uint32_t num_threads = m_nof_buckets_to_compute;
    const uint32_t num_threads_per_group = 256;

    auto pipelineState = METAL_GET_PIPELINE(IsG2 ? "accumulate_buckets_kernel_g2" : "accumulate_buckets_kernel");

    const int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(m_buckets_buffer, 0, 0);
    encoder->setBuffer(m_bucket_offsets_buffer, m_zero_bucket_offset * sizeof(unsigned), 1);
    encoder->setBuffer(m_bucket_sizes_buffer, m_zero_bucket_offset * sizeof(unsigned), 2);
    encoder->setBuffer(m_single_bucket_indices_buffer, m_zero_bucket_offset * sizeof(unsigned), 3);
    encoder->setBuffer(m_point_indices_buffer, 0, 4);
    encoder->setBuffer(m_points_buffer, m_points_offset, 5);
    uint32_t nof_buckets_plus_bms = m_nof_buckets + m_nof_bms_per_msm;
    encoder->setBytes(&nof_buckets_plus_bms, sizeof(uint32_t), 6);
    encoder->setBytes(&m_nof_buckets_to_compute, sizeof(uint32_t), 7);
    uint32_t c_plus_bitsize = m_c + m_bm_bitsize;
    encoder->setBytes(&c_plus_bitsize, sizeof(uint32_t), 8);
    encoder->setBytes(&m_c, sizeof(uint32_t), 9);
    const bool t = true;
    encoder->setBytes(&(t), sizeof(bool), 10);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    m_point_indices_buffer->release();
    m_bucket_indices_buffer->release();
    m_bucket_sizes_buffer->release();
    m_bucket_offsets_buffer->release();
    m_single_bucket_indices_buffer->release();

    return eIcicleError::SUCCESS;
  }

  eIcicleError initialize_projective_buffer(MTL::Buffer* buffer, uint32_t num_points)
  {
    auto pipelineState = METAL_GET_PIPELINE(IsG2 ? "initialize_buckets_kernel_g2" : "initialize_buckets_kernel");
    const uint32_t num_threads = num_points;
    const uint32_t num_threads_per_group = 256;
    const int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(buffer, 0, 0);
    encoder->setBytes(&num_threads, sizeof(uint32_t), 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted(); // TODO do we need to wait here?
    return eIcicleError::SUCCESS;
  }

  void single_stage_multi_reduction(MTL::Buffer* inp, MTL::Buffer* out, const SingleStageMultiReductionParams params)
  {
    auto pipelineState =
      METAL_GET_PIPELINE(IsG2 ? "single_stage_multi_reduction_kernel_g2" : "single_stage_multi_reduction_kernel");
    const uint32_t num_threads_per_group = 256;
    const int nof_blocks = (params.nof_threads + num_threads_per_group - 1) / num_threads_per_group;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(inp, 0, 0);
    encoder->setBuffer(out, 0, 1);
    encoder->setBytes(&params, sizeof(SingleStageMultiReductionParams), 2);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted(); // TODO do we need to wait here?
  }

  eIcicleError bucket_reduction()
  {
    uint32_t nof_final_results_per_msm =
      m_nof_bms_per_msm; // for big-triangle accumluation this is the number of bucket modules

    MTL::Buffer* final_results = nullptr;
    size_t final_results_offset = 0;
    if (m_c == 1) {
      ICICLE_LOG_INFO << "Metal MSM doesn't support m_c == 1 yet";
      // TODO implement big triangle sum
      return eIcicleError::API_NOT_IMPLEMENTED;
    } else {
      uint32_t source_bits_count = m_c;
      uint32_t source_windows_count = m_nof_bms_per_msm;
      uint32_t source_buckets_count = m_nof_buckets + m_nof_bms_per_msm; // nof buckets per msm including zero buckets
      uint32_t target_windows_count;

      auto source_buckets = m_buckets_buffer;
      MTL::Buffer* target_buckets = nullptr;
      MTL::Buffer* temp_buckets1 = nullptr;
      MTL::Buffer* temp_buckets2 = nullptr;

      for (uint32_t i = 0;; i++) {
        const uint32_t target_bits_count = (source_bits_count + 1) >> 1;                 // half the bits rounded up
        target_windows_count = source_windows_count << 1;                                // twice the number of bms
        const uint32_t target_buckets_count = target_windows_count << target_bits_count; // new_bms*2^new_c

        ICICLE_LOG_DEBUG << "Source bits count: " << source_bits_count << "\n"
                         << "Source windows count: " << source_windows_count << "\n"
                         << "Source buckets count: " << source_buckets_count << "\n"
                         << "Target windows count: " << target_windows_count << "\n"
                         << "Target buckets count: " << target_buckets_count << "\n"
                         << "Target bits count: " << target_bits_count << "\n";

        std::tie(target_buckets, m_dummy_offset) =
          metal::allocate_on_device_private(sizeof(P) * target_buckets_count * m_config.batch_size);

        // for type1 reduction (strided, bottom window - evens)
        std::tie(temp_buckets1, m_dummy_offset) =
          metal::allocate_on_device_private(sizeof(P) * source_buckets_count * m_config.batch_size);
        // for type2 reduction (serial, top window - odds)
        std::tie(temp_buckets2, m_dummy_offset) =
          metal::allocate_on_device_private(sizeof(P) * source_buckets_count * m_config.batch_size);

        // initialization is needed for the odd c case
        initialize_projective_buffer(target_buckets, target_buckets_count * m_config.batch_size);

        for (uint32_t j = 0; j < target_bits_count; j++) {
          const bool is_first_iter = (j == 0);
          const bool is_second_iter = (j == 1);
          const bool is_last_iter = (j == target_bits_count - 1);
          const bool is_odd_c = source_bits_count & 1;

          if (!is_odd_c || !is_first_iter) { // skip if c is odd and it's the first iteration

            const uint32_t num_threads =
              (((source_windows_count << target_bits_count) - source_windows_count) << (target_bits_count - 1 - j)) *
              m_config.batch_size; // nof sections to reduce (minus the section that goes to zero buckets) shifted by
                                   // nof threads per section
            const SingleStageMultiReductionParams params = {
              1u << source_bits_count,                            /* orig_block_size */
              1u << (source_bits_count - j + (is_odd_c ? 1 : 0)), /* block_size */
              is_last_iter ? 1u << target_bits_count : 0u,        /* write_stride */
              1u << target_bits_count,                            /* buckets_per_bm */
              0u,                                                 /* write_phase */
              (1u << target_bits_count) - 1,                      /* step */
              num_threads                                         /* nof_threads */
            };

            single_stage_multi_reduction(
              is_first_iter || (is_second_iter && is_odd_c) ? source_buckets : temp_buckets1,
              is_last_iter ? target_buckets : temp_buckets1, params);
          }

          const uint32_t num_threads =
            (((source_windows_count << (source_bits_count - target_bits_count)) - source_windows_count)
             << (target_bits_count - 1 - j)) *
            m_config.batch_size; // nof sections to reduce (minus the section that goes to zero buckets) shifted by
                                 // nof threads per section
          const SingleStageMultiReductionParams params = {
            1u << target_bits_count,                              /* orig_block_size */
            1u << (target_bits_count - j),                        /* block_size */
            is_last_iter ? 1u << target_bits_count : 0u,          /* write_stride */
            1u << (target_bits_count - (is_odd_c ? 1 : 0)),       /* buckets_per_bm */
            1u,                                                   /* write_phase */
            (1u << (target_bits_count - (is_odd_c ? 1 : 0))) - 1, /* step */
            num_threads                                           /* nof_threads */
          };

          single_stage_multi_reduction(
            is_first_iter ? source_buckets : temp_buckets2, is_last_iter ? target_buckets : temp_buckets2, params);
        }

        if (target_bits_count == 1) {
          // Note: the reduction ends up with 'target_windows_count' windows per batch element. Some are guaranteed
          // to be empty when target_windows_count>bitsize. for example consider bitsize=253 and c=2. The reduction
          // ends with 254 bms but the most significant one is guaranteed to be zero since the scalars are 253b.
          // precomputation and odd c can cause additional empty windows.
          nof_final_results_per_msm = std::min(m_c * m_nof_bms_per_msm, m_bitsize);
          m_nof_bms_per_msm = target_windows_count;
          uint32_t total_nof_final_results = nof_final_results_per_msm * m_config.batch_size;

          std::tie(final_results, final_results_offset) =
            metal::allocate_on_device_private(sizeof(P) * total_nof_final_results);

          auto pipelineState = METAL_GET_PIPELINE(IsG2 ? "last_pass_kernel_g2" : "last_pass_kernel");
          const uint32_t num_threads = total_nof_final_results;
          const uint32_t num_threads_per_group = 32;
          const int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
          MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
          MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

          MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
          MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
          encoder->setComputePipelineState(pipelineState);
          encoder->setBuffer(target_buckets, 0, 0);
          encoder->setBuffer(final_results, 0, 1);
          encoder->setBytes(&(nof_final_results_per_msm), sizeof(uint32_t), 2);
          encoder->setBytes(&(m_config.batch_size), sizeof(uint32_t), 3);
          encoder->setBytes(&(m_nof_bms_per_msm), sizeof(uint32_t), 4);
          encoder->setBytes(&(m_c), sizeof(uint32_t), 5);
          encoder->dispatchThreadgroups(gridSize, threadGroupSize);
          encoder->endEncoding();
          commandBuffer->commit();
          commandBuffer->waitUntilCompleted(); // TODO do we need to wait here?

          source_buckets->release();
          target_buckets->release();
          temp_buckets1->release();
          temp_buckets2->release();
          break;
        }
        source_buckets->release();
        temp_buckets1->release();
        temp_buckets2->release();
        source_buckets = target_buckets;
        source_bits_count = target_bits_count;
        source_windows_count = target_windows_count;
        source_buckets_count = target_buckets_count;
      }
    }

    // ------- This is the final stage where bucket modules/window sums get added up with appropriate weights
    // -------
    // launch the double and add kernel, a single thread per batch element
    auto pipelineState = METAL_GET_PIPELINE(IsG2 ? "final_accumulation_kernel_g2" : "final_accumulation_kernel");
    const uint32_t num_threads = m_config.batch_size;
    const uint32_t num_threads_per_group = 32;
    const int nof_blocks = (num_threads + num_threads_per_group - 1) / num_threads_per_group;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(num_threads_per_group, 1, 1);

    MTL::CommandBuffer* commandBuffer = m_cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(final_results, final_results_offset, 0);
    encoder->setBuffer(m_final_result, 0, 1);
    encoder->setBytes(&(m_config.batch_size), sizeof(uint32_t), 2);
    encoder->setBytes(&(nof_final_results_per_msm), sizeof(uint32_t), 3);
    const uint32_t temp_c = 1;
    encoder->setBytes(&(temp_c), sizeof(uint32_t), 4);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted(); // TODO do we need to wait here?

    final_results->release();

    return eIcicleError::SUCCESS;
  }

  // this function computes msm using the bucket method
  eIcicleError bucket_method_msm(const S* scalars, const A* points, P* final_result, bool is_async)
  {
    // (1) Copy scalars to GPU if on host and convert from montgomery if need to
    ICICLE_LOG_DEBUG << "MSM Upload scalars";
    upload_scalars(scalars);

    // (2) upload points to device in parallel to scalar sorting (using another stream)
    ICICLE_LOG_DEBUG << "MSM Upload points";
    upload_points(points);

    // (3) Split and sort scalars
    split_and_sort_scalars();

    // (4) Allocate and sort buckets by size
    buckets_init_and_sort();

    // (5) Accumulate buckets
    bucket_accumulation();

    std::tie(m_final_result, m_dummy_offset) =
      m_config.are_results_on_device
        ? metal::map_to_metal_buffer(final_result)
        : metal::allocate_on_device(
            m_config.batch_size * sizeof(P)); // TODO - would it be better to use private buffer?

    // (6) Bucket Reduction - sum each bucket module then finally sum all buckets module to final sum
    bucket_reduction();

    if (!m_config.is_async || !m_config.are_results_on_device) { metal::synchronize_stream(m_config.stream); }
    if (!m_config.are_results_on_device) {
      memcpy(final_result, m_final_result->contents(), m_config.batch_size * sizeof(P));
      m_final_result->release();
    }

    return eIcicleError::SUCCESS;
  }
};

template <typename S, typename A, typename P>
eIcicleError
msm_metal(const Device& device, const S* scalars, const A* points, int msm_size, const MSMConfig& config, P* results)
{
  // TODO - currently we don't support batch with not shared points, remove this check when we do
  if (config.are_points_shared_in_batch == false && config.batch_size > 1) {
    ICICLE_LOG_WARNING << "Metal MSM currently does not support batch with not shared points";
    return eIcicleError::API_NOT_IMPLEMENTED;
  }
  MetalMSMBackend<S, P, A> msm_engine{msm_size, config};
  msm_engine.bucket_method_msm(scalars, points, results, false);
  return eIcicleError::SUCCESS;
}

template <typename S, typename A, typename P>
eIcicleError metal_msm_precompute_bases(
  const Device& device, const A* input_bases, int nof_bases, const MSMConfig& config, A* output_bases)
{
  if (config.precompute_factor > 1) {
    ICICLE_LOG_WARNING << "Metal MSM currently does not support precompute factor bigger than 1";
  }
  memcpy(output_bases, input_bases, nof_bases * sizeof(A));
  return eIcicleError::SUCCESS;
}