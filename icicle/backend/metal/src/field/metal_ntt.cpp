#include "Metal/Metal.h"

#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "metal_device_api.h"
#include "metal_library_loader.h"

using namespace field_config;
using namespace icicle;

// class MetalNttDomain is a singleton class that manages the NTT domain and twiddles for the Metal backend.
template <typename S = scalar_t>
class MetalNttDomain
{
public:
  static MetalNttDomain& get_instance()
  {
    static MetalNttDomain instance;
    return instance;
  }

  eIcicleError init(const scalar_t& primitive_root, const NTTInitDomainConfig& config)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    const bool is_initialized = m_twiddles != nullptr;
    if (is_initialized) {
      ICICLE_LOG_WARNING << "NTT domain already initialized with domain-log-size " << m_domain_log_size
                         << " for primitive root=" << m_primitive_root
                         << ". Skipping re-initialization. To re-initialize, call release() first.";
      return eIcicleError::SUCCESS;
    }

    m_domain_log_size = 0;
    bool found_logn = false;
    S omega = primitive_root;
    const unsigned omegas_count = S::get_omegas_count();
    for (int i = 0; i < omegas_count; i++) {
      omega = omega.sqr();
      if (!found_logn) {
        ++m_domain_log_size;
        found_logn = omega == S::one();
        if (found_logn) break;
      }
    }

    if (omega != S::one()) {
      ICICLE_LOG_ERROR << "Primitive root provided to the InitDomain function is not a root-of-unity";
      return eIcicleError::INVALID_ARGUMENT;
    }

    m_primitive_root = primitive_root;

    // +1 is for additional ONE element at the end for efficient inverse NTT
    const uint32_t domain_size = (1 << m_domain_log_size) + 1;
    // Note: twiddles are allocate as private buffer since only GPU needs to access it
    auto [twiddlesBuffer, offset] = metal::allocate_on_device_private(domain_size * sizeof(S));

    // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
    MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                             : metal::get_default_command_queue();

    auto pipelineState = METAL_GET_PIPELINE("ntt_init_domain");

    MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(twiddlesBuffer, offset, 0);
    encoder->setBytes(&primitive_root, sizeof(primitive_root), 1);
    encoder->setBytes(&domain_size, sizeof(domain_size) /*length*/, 2 /*index*/);

    const int NOF_THREADS_PER_BLOCK = 256;
    const int NOF_BLOCKS = (domain_size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
    MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();

    // Construct coset-idx map: it maps a coset generator to its index in the domain
    m_m_coset_index_map.clear();
    m_m_coset_index_map[S::one()] = 0;
    for (int i = 0; i < m_domain_log_size; ++i) {
      const int index = 1 << i;                    // 2^i
      const S twiddle = primitive_root.pow(index); // primitive_root^index
      m_m_coset_index_map[twiddle] = index;        // map w^index to index
    }

    m_twiddles = twiddlesBuffer;
    if (config.is_async) { return eIcicleError::SUCCESS; }

    // For synchronous case: wait for completion, copy output back to host and release allocated resources
    ICICLE_ASSERT(!config.is_async);
    commandBuffer->waitUntilCompleted();

    return eIcicleError::SUCCESS;
  }

  eIcicleError release()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_twiddles != nullptr) {
      m_twiddles->release();
      m_twiddles = nullptr;
    }
    m_domain_log_size = 0;
    m_m_coset_index_map.clear();
    return eIcicleError::SUCCESS;
  }

  std::tuple<MTL::Buffer*, uint32_t /*offset*/, uint32_t /*nof_twiddles*/> get_twiddles() const
  {
    return std::make_tuple(m_twiddles, 0, 1 << m_domain_log_size);
  }

  eIcicleError get_root_of_unity_from_domain(uint64_t logn, S* rou)
  {
    if (nullptr == m_twiddles) {
      ICICLE_LOG_ERROR << "NTT domain not initialized";
      return eIcicleError::UNKNOWN_ERROR;
    }

    if (logn > m_domain_log_size) {
      ICICLE_LOG_ERROR << "NTT log_size=" << logn << " is too large for the domain (logsize=" << m_domain_log_size
                       << "). Consider generating your domain with a higher order root of unity.\n ";
      return eIcicleError::INVALID_ARGUMENT;
    }
    *rou = m_primitive_root.pow(1 << (m_domain_log_size - logn));
    return eIcicleError::SUCCESS;
  }

  const std::unordered_map<S, uint32_t>& get_m_coset_index_map() const { return m_m_coset_index_map; }

private:
  MetalNttDomain() = default;
  ~MetalNttDomain() { release(); }

  MetalNttDomain(const MetalNttDomain&) = delete;
  MetalNttDomain& operator=(const MetalNttDomain&) = delete;

  S m_primitive_root;
  uint32_t m_domain_log_size = 0;
  MTL::Buffer* m_twiddles = nullptr;
  std::unordered_map<S, uint32_t> m_m_coset_index_map; // maps coset generator to its index in twiddles
  std::mutex m_mutex;
};

/*====================================================================================================================================================================*/
// class MetalNttBackend is a backend for the NTT operation using Metal
template <typename S, typename E>
class MetalNttBackend
{
private:
  const uint32_t m_size;
  const NTTConfig<S>& m_config;
  // Derived constants
  const uint32_t m_logn;
  const uint32_t m_inverse;
  const uint32_t m_batch_size;
  const uint32_t m_column_batch;
  const uint64_t m_total_size; // size * batch_size
  uint32_t m_coset_index = 0;

  // Metal buffers
  MTL::Buffer* m_input_buffer = nullptr;
  size_t m_input_offset = 0;
  MTL::Buffer* m_output_buffer = nullptr;
  size_t m_output_offset = 0;
  MTL::Buffer* twiddles = nullptr;
  size_t m_twiddle_offset = 0;
  uint32_t m_twiddle_count = 0;
  bool m_release_input_buffer =
    false; // Currently unused since ntt inplace needs only output buffer so never allocating input
  bool m_release_output_buffer = false;

public:
  MetalNttBackend(uint32_t size, NTTDir dir, const NTTConfig<S>& config)
      : m_size(size), m_config(config), m_logn(log2(size)), m_inverse(dir == NTTDir::kInverse),
        m_column_batch(config.columns_batch), m_batch_size(config.batch_size),
        m_total_size(static_cast<uint64_t>(size) * config.batch_size)
  // Assuming size is a power of 2 (Checked in the caller)
  {
  }

  ~MetalNttBackend()
  {
    if (m_release_input_buffer && m_input_buffer) { m_input_buffer->release(); }
    if (m_release_output_buffer && m_output_buffer) { m_output_buffer->release(); }
  }

  eIcicleError compute_ntt(const E* input, E* output)
  {
    eIcicleError err = init_and_verify_params();
    if (err != eIcicleError::SUCCESS) { return err; }

    err = map_or_allocate_io_buffers(input, output);
    if (err != eIcicleError::SUCCESS) { return err; }

    auto [dit, reverse_input] = determine_ordering();

    if (reverse_input) { reverse_input_batch(/*optional: threads_per_block*/); }

    // Note: cosets are evaluated by a preprocess multiplication by twiddle followed by NTT.
    // For inverse it's postprocessing

    const bool apply_coset_multiplication = m_coset_index != 0;
    if (apply_coset_multiplication && !m_inverse) { coset_mul_batch(dit /*, optional: threads_per_block*/); }

    radix2_ntt_loop(dit /*, optional: threads_per_block*/);

    if (apply_coset_multiplication && m_inverse) { coset_mul_batch(dit /*, optional: threads_per_block*/); }

    if (m_inverse) { normalize_output_buffer(); }

    // If host is on host, must synchronize before copying the output data from a metal buffer
    if (!m_config.is_async || !m_config.are_outputs_on_device) { metal::synchronize_stream(m_config.stream); }
    if (!m_config.are_outputs_on_device) { std::memcpy(output, m_output_buffer->contents(), m_size * sizeof(E)); }

    return eIcicleError::SUCCESS;
  }

  // Init handles things that may go wrong, to avoid exceptions in the constructor
  eIcicleError init_and_verify_params()
  {
    const bool size_is_power_of_two = (m_size & (m_size - 1)) == 0;
    if (!size_is_power_of_two) {
      ICICLE_LOG_ERROR << "Size must be a power of 2";
      return eIcicleError::INVALID_ARGUMENT;
    }

    // Metal NTT supports only sizes up to 2^32 (including full batch)
    if (m_total_size > UINT32_MAX) {
      ICICLE_LOG_ERROR << "Metal NTT supports input sizes up to 2^32 (including full batch). "
                       << "Requested: size=" << m_size << ", batch=" << m_config.batch_size
                       << " (total=" << m_total_size << "). "
                       << "Please retry with a smaller batch size or fallback to the CPU implementation.";
      return eIcicleError::INVALID_ARGUMENT;
    }

    m_coset_index = 0;
    if (m_config.coset_gen != S::one()) {
      auto m_coset_index_map = MetalNttDomain<S>::get_instance().get_m_coset_index_map();
      auto m_coset_index_it = m_coset_index_map.find(m_config.coset_gen);
      if (m_coset_index_map.end() == m_coset_index_it) {
        ICICLE_LOG_ERROR << "Coset generator not found in the domain. Arbitrary coset not supported in METAL backend";
        return eIcicleError::INVALID_ARGUMENT;
      }
      m_coset_index = m_coset_index_it->second;
    }

    return eIcicleError::SUCCESS;
  }

  eIcicleError map_or_allocate_io_buffers(const E* input, E* output)
  {
    std::tie(twiddles, m_twiddle_offset, m_twiddle_count) = MetalNttDomain<S>::get_instance().get_twiddles();
    if (nullptr == twiddles) {
      ICICLE_LOG_ERROR << "NTT domain not initialized";
      return eIcicleError::UNKNOWN_ERROR;
    }

    // If input is on host, we must synchronize before copying the input data to a metal buffer (assuming same stream,
    // otherwise user is responsible to synchronize)
    if (!m_config.are_inputs_on_device) { metal::synchronize_stream(m_config.stream); }

    // Input/oputput buffers handling
    // match 2'b{input_on_device, output_on_device}:
    switch ((m_config.are_inputs_on_device << 1) | m_config.are_outputs_on_device) {
    case 0b11: {
      //   {true, true} => first stage from input to output and then in-place
      std::tie(m_input_buffer, m_input_offset) = metal::map_to_metal_buffer(input);
      std::tie(m_output_buffer, m_output_offset) = metal::map_to_metal_buffer(output);
      break;
    }
    case 0b10: {
      //   {true, false} => allocate metal buffer, first stage from input to buffer and then in-place
      std::tie(m_input_buffer, m_input_offset) = metal::map_to_metal_buffer(input);
      std::tie(m_output_buffer, m_output_offset) = metal::allocate_on_device(m_size * sizeof(E));
      m_release_output_buffer = true;
      break;
    }
    case 0b01: {
      //   {false, true} => copy input to output and all the way in place
      std::tie(m_output_buffer, m_output_offset) = metal::map_to_metal_buffer(output);
      std::tie(m_input_buffer, m_input_offset) = std::make_pair(m_output_buffer, m_output_offset);
      std::memcpy(m_output_buffer->contents(), input, m_size * sizeof(E));
      break;
    }
    case 0b00: {
      //   {false, false} => allocate metal buffer, copy input to buffer, all the way inplace and copy back to output
      std::tie(m_output_buffer, m_output_offset) = metal::allocate_and_copy_to_device(input, m_size * sizeof(E));
      std::memcpy(m_output_buffer->contents(), input, m_size * sizeof(E));
      std::tie(m_input_buffer, m_input_offset) = std::make_pair(m_output_buffer, m_output_offset);
      m_release_output_buffer = true;
      break;
    }
    }

    return eIcicleError::SUCCESS;
  }

  std::pair<uint32_t /*dit*/, bool /*reverse_input*/> determine_ordering() const
  {
    uint32_t dit = 1; // avoiding bool since they are 4B in metal anyway
    bool reverse_input = false;
    switch (m_config.ordering) {
    case Ordering::kNN:
      reverse_input = true;
      break;
    case Ordering::kNR:
    case Ordering::kNM:
      dit = 0;
      break;
    case Ordering::kRR:
      reverse_input = true;
      dit = 0;
      break;
    case Ordering::kRN:
    case Ordering::kMN:
      reverse_input = false;
    }
    return std::make_pair(dit, reverse_input);
  }

  void reverse_input_batch(int nof_threads_per_block = 256)
  {
    MTL::CommandQueue* cmdQueue = (m_config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(m_config.stream)
                                                               : metal::get_default_command_queue();
    auto pipelineStateReverseInput = METAL_GET_PIPELINE("ntt_reverse_order");
    MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineStateReverseInput);
    encoder->setBuffer(m_input_buffer, m_input_offset, 0);
    encoder->setBuffer(m_output_buffer, m_output_offset, 1);
    encoder->setBytes(&m_size, sizeof(uint32_t), 2);
    encoder->setBytes(&m_logn, sizeof(uint32_t), 3);
    encoder->setBytes(&m_batch_size, sizeof(uint32_t), 4);
    encoder->setBytes(&m_column_batch, sizeof(uint32_t), 5);

    const int nof_blocks = (m_total_size + nof_threads_per_block - 1) / nof_threads_per_block;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(nof_threads_per_block, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();

    // update input buffer to be the output buffer
    m_input_buffer = m_output_buffer;
    m_input_offset = m_output_offset;
  }

  void normalize_output_buffer(int nof_threads_per_block = 256)
  {
    MTL::CommandQueue* cmdQueue = (m_config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(m_config.stream)
                                                               : metal::get_default_command_queue();
    const S inv_n = S::inv_log_size(m_logn);

    auto pipelineStateNormalize = METAL_GET_PIPELINE("ntt_normalize");
    MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineStateNormalize);
    encoder->setBuffer(m_output_buffer, m_output_offset, 0);
    encoder->setBytes(&inv_n, sizeof(S), 1);
    encoder->setBytes(&m_total_size, sizeof(uint32_t), 2);

    const int nof_blocks = (m_total_size + nof_threads_per_block - 1) / nof_threads_per_block;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(nof_threads_per_block, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
  }

  void coset_mul_batch(uint32_t dit, int nof_threads_per_block = 256)
  {
    // Note: this method behaves a little different for inverse NTT so need to be careful where calling it
    // For forward, it's a preprocess step, but for inverse it's a post process step

    const uint32_t dif = !dit;
    MTL::CommandQueue* cmdQueue = (m_config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(m_config.stream)
                                                               : metal::get_default_command_queue();

    auto pipelineStateCosetMultiplication = METAL_GET_PIPELINE("ntt_batch_mul");
    MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineStateCosetMultiplication);
    encoder->setBuffer(m_input_buffer, m_input_offset, 0);
    encoder->setBuffer(twiddles, m_twiddle_offset, 1);
    encoder->setBuffer(m_output_buffer, m_output_offset, 2);
    encoder->setBytes(&m_size, sizeof(uint32_t), 3);
    encoder->setBytes(&m_batch_size, sizeof(uint32_t), 4);
    encoder->setBytes(&m_column_batch, sizeof(uint32_t), 5);
    encoder->setBytes(&m_coset_index, sizeof(uint32_t), 6);
    encoder->setBytes(&m_twiddle_count, sizeof(uint32_t), 7);
    encoder->setBytes(&m_logn, sizeof(uint32_t), 8);
    encoder->setBytes(m_inverse ? &dif : &dit, sizeof(uint32_t), 9);
    encoder->setBytes(&m_inverse, sizeof(uint32_t), 10);

    const int nof_blocks = (m_total_size + nof_threads_per_block - 1) / nof_threads_per_block;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(nof_threads_per_block, 1, 1);
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();

    // update input buffer to be the output buffer
    m_input_buffer = m_output_buffer;
    m_input_offset = m_output_offset;
  }

  void radix2_ntt_loop(uint32_t dit, int nof_threads_per_block = 256)
  {
    // Note: the kernel is designed to handle 2 elements per thread but can be easily modified
    const uint32_t num_threads = m_total_size / 2;

    MTL::CommandQueue* cmdQueue = (m_config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(m_config.stream)
                                                               : metal::get_default_command_queue();
    auto pipelineStateRadix2Stage = METAL_GET_PIPELINE("ntt_stage_radix2");

    const int nof_blocks = (num_threads + nof_threads_per_block - 1) / nof_threads_per_block;
    MTL::Size gridSize = MTL::Size(nof_blocks, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(nof_threads_per_block, 1, 1);

    const int nof_stages = m_logn;
    for (int stage_idx = 1; stage_idx <= nof_stages; ++stage_idx) {
      const bool is_first_stage = stage_idx == 1;

      const int stage = dit ? stage_idx : nof_stages - stage_idx + 1;

      MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
      MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

      encoder->setComputePipelineState(pipelineStateRadix2Stage);
      encoder->setBuffer(
        is_first_stage ? m_input_buffer : m_output_buffer, is_first_stage ? m_input_offset : m_output_offset, 0);
      encoder->setBuffer(m_output_buffer, m_output_offset, 1);
      encoder->setBuffer(twiddles, m_twiddle_offset, 2);
      encoder->setBytes(&m_size, sizeof(uint32_t), 3);
      encoder->setBytes(&stage, sizeof(uint32_t), 4);
      encoder->setBytes(&m_twiddle_count, sizeof(int32_t), 5);
      encoder->setBytes(&num_threads, sizeof(int32_t), 6);
      encoder->setBytes(&dit, sizeof(uint32_t), 7);
      encoder->setBytes(&m_inverse, sizeof(uint32_t), 8);
      encoder->setBytes(&m_batch_size, sizeof(uint32_t), 9);
      encoder->setBytes(&m_column_batch, sizeof(uint32_t), 10);

      encoder->dispatchThreadgroups(gridSize, threadGroupSize);

      encoder->endEncoding();
      commandBuffer->commit();
    }
  }
};

template <typename S = scalar_t>
eIcicleError metal_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config)
{
  return MetalNttDomain<S>::get_instance().init(primitive_root, config);
}

template <typename S = scalar_t>
eIcicleError metal_ntt_release_domain(const Device& device, const S& dummy)
{
  return MetalNttDomain<S>::get_instance().release();
}

template <typename S = scalar_t>
eIcicleError metal_get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou)
{
  return MetalNttDomain<S>::get_instance().get_root_of_unity_from_domain(logn, rou);
}

template <typename S, typename E>
eIcicleError
metal_ntt(const Device& device, const E* input, uint32_t size, NTTDir dir, const NTTConfig<S>& config, E* output)
{
  // Note: this NTT object is stateful and the state is modified by methods during the computation.
  // This is a design choice to avoid passing arguments everywhere.

  MetalNttBackend<S, E> ntt_engine{size, dir, config};
  return ntt_engine.compute_ntt(input, output);
}

REGISTER_NTT_INIT_DOMAIN_BACKEND("METAL", (metal_ntt_init_domain<scalar_t>));
REGISTER_NTT_RELEASE_DOMAIN_BACKEND("METAL", metal_ntt_release_domain<scalar_t>);
REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND("METAL", metal_get_root_of_unity_from_domain<scalar_t>);
REGISTER_NTT_BACKEND("METAL", (metal_ntt<scalar_t, scalar_t>));

#ifdef EXT_FIELD
// REGISTER_NTT_EXT_FIELD_BACKEND("METAL", (metal_ntt<scalar_t, extension_t>));
#endif // EXT_FIELD