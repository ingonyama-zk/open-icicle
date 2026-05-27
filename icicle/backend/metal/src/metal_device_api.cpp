
#include "Metal/Metal.h"

#include "icicle/device_api.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/runtime.h"
#include "icicle/memory_tracker.h"

namespace icicle {

  class MetalDeviceAPI : public DeviceAPI
  {
  public: // not accessible outside of this compilation unit anyway
    static inline std::mutex s_initialize_mutex;
    static inline MTL::Device* s_metal_device = nullptr;
    // This is the default stream and it is thread-safe by design (of Metal)
    static inline MTL::CommandQueue* s_default_command_queue = nullptr;
    // Memory tracking to map addresses to Metal buffers and their offsets.
    // Required for operations that need access to the Metal buffer (e.g., kernel dispatches, async operations).
    static inline MemoryTracker<MTL::Buffer*> s_buffer_tracker;

    static eIcicleError initialize()
    {
      if (nullptr != s_metal_device) {
        return eIcicleError::SUCCESS; // already initialized
      }

      // Lock to ensure thread safety
      std::lock_guard<std::mutex> lock(s_initialize_mutex);
      // double check
      if (nullptr != s_metal_device) {
        return eIcicleError::SUCCESS; // already initialized
      }

      // Create an autorelease pool to manage memory for objects marked as autorelease.
      // This is required because Metal and Cocoa APIs often return autoreleased objects.
      // The pool ensures that any autoreleased objects created within its scope are properly cleaned up.
      NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();

      // Create the default Metal device for the system.
      // This initializes the Metal framework and retrieves the primary GPU device to be used for rendering or
      // computation. The returned device object is managed by the autorelease pool created above.
      s_metal_device = MTL::CreateSystemDefaultDevice();
      if (nullptr == s_metal_device) {
        ICICLE_LOG_ERROR << "Metal is not supported on this device";
        return eIcicleError::INVALID_DEVICE;
      }

      // Check for unified memory support (may relax this requirement if needed)
      if (!s_metal_device->hasUnifiedMemory()) {
        ICICLE_LOG_ERROR << "Metal is not supported on this device due to missing support: Unified-Memory";
        s_metal_device = nullptr;
        return eIcicleError::INVALID_DEVICE;
      }

      s_default_command_queue = s_metal_device->newCommandQueue();
      if (nullptr == s_default_command_queue) {
        ICICLE_LOG_ERROR << "Failed to create command-queue for Metal device";
        return eIcicleError::STREAM_CREATION_FAILED;
      }

      ICICLE_LOG_INFO << "Metal device '" << s_metal_device->name()->cString(NS::StringEncoding::UTF8StringEncoding)
                      << "' initialized successfully.";

      return eIcicleError::SUCCESS;
    }

    eIcicleError set_device(const Device& device) override
    {
      // Note: try_checkout_device(id) will generate a challenge and expect the license logic (in device) to know how to
      // respond. If it doesn't then the check fails. This is making sure that a real cuda_device lib is involved. Can
      // be hacked via man in the middle attack probably but should make it harder to mock this cuda device lib (which
      // is otherwise easy to implement without license and mock the license check function.)
      if (!try_checkout_device(device.id)) {
        Device active_device = "";
        ICICLE_CHECK(icicle_get_active_device(active_device));
        ICICLE_LOG_ERROR << "Failed to checkout license for device " << device << ". Current device remains "
                         << active_device;
        return eIcicleError::LICENSE_CHECK_ERROR;
      }
      // Note that assuming single device for Apple-silicon
      if (device.id != 0) {
        ICICLE_LOG_ERROR << "Invalid devic-id (=" << device.id << ") for Metal backend. Only device-id=0 is available";
        return eIcicleError::INVALID_DEVICE;
      }
      return initialize();
    }

    static MTL::Device* get_device()
    {
      if (eIcicleError::SUCCESS != initialize()) return nullptr;
      return s_metal_device;
    }

    static MTL::CommandQueue* get_default_command_queue()
    {
      if (eIcicleError::SUCCESS != initialize()) return nullptr;
      return s_default_command_queue;
    }

    eIcicleError get_device_count(int& device_count) const override
    {
      device_count = 1; // single logical device for Apple-silicon devices, including M2-'Ultra' and similar
      return eIcicleError::SUCCESS;
    }

    // Memory management

    // Tracker is used to track addresses to {Metal buffer, offset}
    // Note that it must support mapping addresses with offset to allocation. For example |start....^.....end| where we
    // get '^' as address
    static void track_memory(void* ptr, size_t size, MTL::Buffer* buffer)
    {
      s_buffer_tracker.add_allocation(ptr, size, buffer);
    }
    static void untrack_memory(void* ptr) { s_buffer_tracker.remove_allocation(ptr); }
    // translate ptr to {Buffer, offset}. Note that ptr can be an offset into an allocation
    static std::optional<std::pair<MTL::Buffer*, size_t>> map_to_metal_buffer(const void* ptr)
    {
      // identify() returns std::optional<std::pair<MTL::Buffer**, size_t>> but I need
      // std::optional<std::pair<MTL::Buffer*, size_t>>.
      if (auto it = s_buffer_tracker.identify(ptr)) { return std::make_pair(*it->first, it->second); }
      return std::nullopt;
    }

    eIcicleError allocate_memory(void** ptr, size_t size) const override
    {
      if (nullptr == s_metal_device) {
        ICICLE_LOG_ERROR << "Metal device not initialized";
        return eIcicleError::INVALID_DEVICE;
      }

      if (size == 0) {
        ICICLE_LOG_ERROR << "Cannot allocate memory of size 0";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // Allocating a shared (CPU-GPU only) buffer since otherwise it is very expensive for CPU to read/write (need an
      // additional shared-buffer copy)
      MTL::Buffer* metal_buffer = s_metal_device->newBuffer(size, MTL::ResourceStorageModeShared);
      if (!metal_buffer) {
        ICICLE_LOG_ERROR << "Failed to allocate Metal buffer of size " << size;
        return eIcicleError::OUT_OF_MEMORY;
      }

      // In order for users to do pointer arithmetic I must give them back the internal pointer.
      // But later I will need to be able to retrieve the MTL::Buffer to release it, or dispatch kernels so must keep
      // record.
      *ptr = metal_buffer->contents();
      track_memory(*ptr, size, metal_buffer);
      return eIcicleError::SUCCESS;
    }

    eIcicleError free_memory(void* ptr) const override
    {
      // free(null) is considered correct behavior in C++
      if (ptr == nullptr) { return eIcicleError::SUCCESS; }

      // Cast the pointer to MTL::Buffer
      auto it = map_to_metal_buffer(ptr);
      if (!it) {
        ICICLE_LOG_ERROR << "Invalid buffer pointer provided for deallocation";
        return eIcicleError::INVALID_DEVICE;
      }

      // Release the Metal buffer
      untrack_memory(ptr);
      MTL::Buffer* buffer = it->first;
      buffer->release();

      return eIcicleError::SUCCESS;
    }

    eIcicleError allocate_memory_async(void** ptr, size_t size, icicleStreamHandle stream) const override
    {
      // Metal buffer allocation is always synchronous
      return allocate_memory(ptr, size);
    }

    eIcicleError free_memory_async(void* ptr, icicleStreamHandle stream) const override
    {
      // Metal buffer release is always synchronous
      return free_memory(ptr);
    }

    // Handles copying between memory regions based on the copy direction and sync/async requirements.
    // For DeviceToDevice copies, we use asynchronous command buffers if requested; for all other directions, use
    // synchronous memcpy since Metal does not support it.
    eIcicleError copy_internal(
      void* dst,
      const void* src,
      size_t size,
      eCopyDirection direction,
      bool is_async,
      MTL::CommandQueue* command_queue) const
    {
      if (dst == nullptr || src == nullptr) {
        ICICLE_LOG_ERROR << "Null pointer provided for copy operation";
        return eIcicleError::INVALID_ARGUMENT;
      }

      if (size == 0) {
        ICICLE_LOG_ERROR << "Copy size cannot be zero";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // DeviceToDevice is the only case that can really be async
      if (is_async && eCopyDirection::DeviceToDevice == direction) {
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        if (!command_buffer) {
          ICICLE_LOG_ERROR << "Failed to create command buffer for GPU-to-GPU copy";
          return eIcicleError::OUT_OF_MEMORY;
        }
        // To use Metal API for copy, we must have the MTL::Buffer and offsets
        auto dst_it = map_to_metal_buffer(dst);
        if (!dst_it) { return eIcicleError::INVALID_POINTER; }
        auto src_it = map_to_metal_buffer(src);
        if (!src_it) { return eIcicleError::INVALID_POINTER; }

        MTL::BlitCommandEncoder* blit_encoder = command_buffer->blitCommandEncoder();
        blit_encoder->copyFromBuffer(src_it->first, src_it->second, dst_it->first, dst_it->second, size);
        blit_encoder->endEncoding();
        command_buffer->commit();
        return eIcicleError::SUCCESS;
      }

      // For other cases (involving host), Metal is always synchronous
      // Since we use shared buffers we can simply use memcpy=
      synchronize_internal(command_queue);
      std::memcpy(dst, src, size);

      return eIcicleError::SUCCESS;
    }

    eIcicleError copy(void* dst, const void* src, size_t size, eCopyDirection direction) const override
    {
      return copy_internal(dst, src, size, direction, false /*=async*/, s_default_command_queue);
    }

    eIcicleError copy_async(
      void* dst, const void* src, size_t size, eCopyDirection direction, icicleStreamHandle stream) const override
    {
      auto command_queue = get_command_queue(stream);
      return copy_internal(dst, src, size, direction, true /*async*/, command_queue);
    }

    eIcicleError get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
    {
      // Placeholder: Metal doesn't provide memory stats directly.
      // Future implementation could use approximate memory based on allocations.
      total = 0; // Unknown
      free = 0;  // Unknown
      return eIcicleError::API_NOT_IMPLEMENTED;
    }

    eIcicleError
    memset_internal(void* ptr, int value, size_t size, bool is_async, MTL::CommandQueue* command_queue) const
    {
      if (ptr == nullptr) {
        ICICLE_LOG_ERROR << "Null pointer provided for memset operation";
        return eIcicleError::INVALID_ARGUMENT;
      }

      if (size == 0) {
        ICICLE_LOG_ERROR << "Memset size cannot be zero";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // for sync copy can simply use memset
      if (!is_async) {
        std::memset(ptr, value, size);
        return eIcicleError::SUCCESS;
      }

      // for async, must use Metal API with MTL::Buffer
      auto it = map_to_metal_buffer(ptr);
      if (!it) {
        ICICLE_LOG_ERROR << "Invalid buffer pointer provided for memset";
        return eIcicleError::INVALID_POINTER;
      }
      auto [metal_buffer, offset] = *it;

      if (!command_queue) {
        ICICLE_LOG_ERROR << "No valid command queue for memset operation";
        return eIcicleError::INVALID_ARGUMENT;
      }

      MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
      if (!command_buffer) {
        ICICLE_LOG_ERROR << "Failed to create command buffer for memset operation";
        return eIcicleError::UNKNOWN_ERROR;
      }

      MTL::BlitCommandEncoder* blit_encoder = command_buffer->blitCommandEncoder();
      if (!blit_encoder) {
        ICICLE_LOG_ERROR << "Failed to create blit encoder for memset operation";
        return eIcicleError::UNKNOWN_ERROR;
      }

      // Use `fillBuffer` to fill the target buffer with the specified byte pattern
      uint8_t pattern = static_cast<uint8_t>(value); // `value` should be within the uint8_t range
      blit_encoder->fillBuffer(metal_buffer, NS::Range(offset, size), value);
      blit_encoder->endEncoding();
      command_buffer->commit();

      return eIcicleError::SUCCESS;
    }

    eIcicleError memset(void* ptr, int value, size_t size) const override
    {
      return memset_internal(ptr, value, size, false /*=async*/, s_default_command_queue);
    }

    eIcicleError memset_async(void* ptr, int value, size_t size, icicleStreamHandle stream) const override
    {
      auto command_queue = get_command_queue(stream);
      return memset_internal(ptr, value, size, true /*=async*/, command_queue);
    }

    eIcicleError synchronize_internal(MTL::CommandQueue* command_queue) const
    {
      // Create a command buffer for synchronization
      MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
      if (!command_buffer) {
        ICICLE_LOG_ERROR << "Failed to create command buffer for synchronization";
        return eIcicleError::OUT_OF_MEMORY;
      }

      // Commit the command buffer and wait for its completion
      command_buffer->commit();
      command_buffer->waitUntilCompleted();

      return eIcicleError::SUCCESS;
    }

    eIcicleError synchronize(icicleStreamHandle stream) const override
    {
      // TODO Yuval: this doesn't synchronize all streams!! How to do it in Metal?
      if (stream == nullptr) { return synchronize_internal(get_default_command_queue()); }

      // Cast the stream handle back to a Metal command queue
      MTL::CommandQueue* command_queue = get_command_queue(stream);
      return synchronize_internal(command_queue);
    }

    // Stream management

    eIcicleError create_stream(icicleStreamHandle* stream) const override
    {
      // a stream is a CommandQueue in Metal
      if (nullptr == s_metal_device) {
        ICICLE_LOG_ERROR << "Metal device not initialized";
        return eIcicleError::INVALID_DEVICE;
      }

      // Create a new command queue to represent the stream
      MTL::CommandQueue* command_queue = s_metal_device->newCommandQueue();
      if (!command_queue) {
        ICICLE_LOG_ERROR << "Failed to create command queue (stream)";
        return eIcicleError::OUT_OF_MEMORY;
      }

      // Store the command queue as the stream handle
      *stream = reinterpret_cast<icicleStreamHandle>(command_queue);

      return eIcicleError::SUCCESS;
    }

    eIcicleError destroy_stream(icicleStreamHandle stream) const override
    {
      if (!stream) {
        ICICLE_LOG_ERROR << "Invalid stream handle";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // Cast the stream handle back to MTL::CommandQueue and release it
      MTL::CommandQueue* command_queue = reinterpret_cast<MTL::CommandQueue*>(stream);
      command_queue->release();

      return eIcicleError::SUCCESS;
    }

    MTL::CommandQueue* get_command_queue(icicleStreamHandle stream) const
    {
      if (stream == nullptr) { return s_default_command_queue; }

      // Cast the stream handle back to a Metal command queue
      return reinterpret_cast<MTL::CommandQueue*>(stream);
    }

    eIcicleError get_device_properties(DeviceProperties& properties) const override
    {
      properties.using_host_memory = false; // since we cannot use memory not allocated by Metal to kernels
      properties.num_memory_regions = 1;
      properties.supports_pinned_memory = false;
      return eIcicleError::SUCCESS;
    }
  };

  namespace metal {
    MTL::Device* get_device() { return MetalDeviceAPI::get_device(); }
    MTL::CommandQueue* get_default_command_queue() { return MetalDeviceAPI::get_default_command_queue(); }

    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_on_device(size_t byte_size)
    {
      MTL::Device* metal_device = get_device();
      MTL::Buffer* metal_buffer = metal_device->newBuffer(byte_size, MTL::ResourceStorageModeShared);
      ICICLE_ASSERT(metal_buffer != nullptr) << "Failed to allocate Metal buffer";
      return {metal_buffer, 0 /*=offset*/};
    }

    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_on_device_private(size_t byte_size)
    {
      MTL::Device* metal_device = get_device();
      MTL::Buffer* metal_buffer = metal_device->newBuffer(byte_size, MTL::ResourceStorageModePrivate);
      ICICLE_ASSERT(metal_buffer != nullptr) << "Failed to allocate Metal buffer";
      return {metal_buffer, 0 /*=offset*/};
    }

    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_and_copy_to_device(const void* host_mem, size_t byte_size)
    {
      MTL::Device* metal_device = get_device();
      MTL::Buffer* metal_buffer = metal_device->newBuffer(
        host_mem, byte_size, MTL::ResourceStorageModeShared,
        ^(void* ptr, NS::UInteger size){
          // Required for zero-copy allocation
          // Do nothing, let user handle release
        });
      ICICLE_ASSERT(metal_buffer != nullptr) << "Failed to allocate Metal buffer";
      return {metal_buffer, 0 /*=offset*/};
    }

    std::pair<MTL::Buffer*, size_t> map_to_metal_buffer(const void* ptr)
    {
      auto metal_buffer_opt = MetalDeviceAPI::map_to_metal_buffer(ptr);
      if (std::nullopt == metal_buffer_opt) {
        ICICLE_LOG_ERROR << "Failed to map ptr to Metal buffer : Invalid address";
        ICICLE_ASSERT(false);
      }

      return *metal_buffer_opt;
    }

    eIcicleError synchronize_stream(icicleStreamHandle stream) { return MetalDeviceAPI().synchronize(stream); }

  } // namespace metal

  REGISTER_DEVICE_API("METAL", MetalDeviceAPI);
} // namespace icicle