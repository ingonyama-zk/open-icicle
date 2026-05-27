#pragma once

#include "Metal/Metal.h"
#include "icicle/errors.h"
#include "icicle/device_api.h"

namespace icicle {
  namespace metal {
    MTL::Device* get_device();
    MTL::CommandQueue* get_default_command_queue();
    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_on_device(size_t byte_size);
    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_on_device_private(size_t byte_size);
    std::pair<MTL::Buffer*, size_t /*offset*/> allocate_and_copy_to_device(const void* host_mem, size_t byte_size);
    std::pair<MTL::Buffer*, size_t /*offset*/> map_to_metal_buffer(const void* ptr);
    eIcicleError synchronize_stream(icicleStreamHandle stream);

    template <typename T>
    static void copyAndPrintBuffer(MTL::Buffer* buffer, std::size_t offset, std::size_t n, const char* buffer_name)
    {
      if (!buffer) {
        ICICLE_LOG_ERROR << "Error: Buffer is null.";
        return;
      }

      // Create a vector and copy elements from the buffer
      std::vector<T> copiedElements(n);
      std::memcpy(copiedElements.data(), buffer->contents(), n * sizeof(T));

      // Print the elements
      for (int i = 0; i < n; i++) {
        ICICLE_LOG_INFO << buffer_name << "[" << i << "] = " << copiedElements[i];
      }
    }
  } // namespace metal

} // namespace icicle