#pragma once

#include "Metal/Metal.h"
#include "metal_lib_config.h"
#include "metal_device_api.h"

#include <unordered_map>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>
#include <dispatch/dispatch.h>

namespace icicle {

// This class wraps a Metal library (.metallib), loads it, retrieves kernels by name,
// and caches the corresponding pipeline objects for efficient reuse.
// It is templated by NAMESPACE, which defines the library's compiled data and name.
// Each backend library is expected to have exactly one instance to avoid contention
// between libraries and to automatically locate compiled kernels.

// To retrieve a kernel, include "metal_lib_config.h" and use the following macro:
#define METAL_GET_PIPELINE(kernel_name) MetalLibrary<METAL_LIB>::instance().get_pipeline(kernel_name);

  template <typename NAMESPACE>
  class MetalLibrary
  {
  public:
    // Singleton access method
    static MetalLibrary& instance()
    {
      static MetalLibrary instance;
      return instance;
    }

    // Retrieve or create a pipeline state for the specified kernel, caching it for future requests
    MTL::ComputePipelineState* get_pipeline(const std::string& kernel_name)
    {
      // Check if the pipeline for this kernel is already cached
      auto pipeline_it = m_pipeline_cache.find(kernel_name);
      if (pipeline_it != m_pipeline_cache.end()) { return pipeline_it->second; }

      // Lock mutex to ensure thread-safe access to the cache and library loading
      std::lock_guard<std::mutex> lock(m_mutex);

      // Double-check the cache after acquiring the lock (for concurrent calls)
      pipeline_it = m_pipeline_cache.find(kernel_name);
      if (pipeline_it != m_pipeline_cache.end()) { return pipeline_it->second; }

      // Construct the fully qualified kernel name as {LIBRARY_NAME}_{KERNEL_NAME}
      auto full_kernel_name =
        NS::String::string((m_library_name + kernel_name).c_str(), NS::StringEncoding::UTF8StringEncoding);

      ICICLE_LOG_VERBOSE << "Trying to load kernel "
                         << full_kernel_name->cString(NS::StringEncoding::UTF8StringEncoding);

      // Load the kernel function from the library
      MTL::Function* function = m_metal_library->newFunction(full_kernel_name);
      if (!function) {
        ICICLE_LOG_ERROR << "Failed to find kernel: "
                         << full_kernel_name->cString(NS::StringEncoding::UTF8StringEncoding)
                         << " in library: " << m_library_name;
        ICICLE_ASSERT(false); // consider this fatal error to avoid all kernel dispatch to handle this case
        return nullptr;
      }

      // Create the pipeline state for the function
      NS::Error* error = nullptr;
      MTL::ComputePipelineState* pipeline_state = m_device->newComputePipelineState(function, &error);
      function->release(); // Release function after creating the pipeline state

      if (!pipeline_state) {
        ICICLE_LOG_ERROR << "Failed to create pipeline state for kernel: " << kernel_name << ": "
                         << error->localizedDescription()->utf8String();
        return nullptr;
      }

      // Cache the newly created pipeline state for future calls and return it
      m_pipeline_cache[kernel_name] = pipeline_state;
      return pipeline_state;
    }

    // Deleted copy and move constructors to enforce singleton pattern
    MetalLibrary(const MetalLibrary&) = delete;
    MetalLibrary(MetalLibrary&&) = delete;
    MetalLibrary& operator=(const MetalLibrary&) = delete;

  private:
    // Constructor that loads a Metal library from embedded data defined in NAMESPACE
    MetalLibrary() : m_device{nullptr}, m_library_name{NAMESPACE::metallib_library_prefix}
    {
      m_device = metal::get_device();
      ICICLE_ASSERT(m_device != nullptr) << "Invalid Metal device";

      NS::Error* error = nullptr;
      dispatch_data_t metallib_dispatch_data = dispatch_data_create(
        NAMESPACE::metallib_data,        // Pointer to the start of the data
        NAMESPACE::metallib_data_len,    // Size of the data in bytes
        NULL,                            // Dispatch queue (NULL for default)
        DISPATCH_DATA_DESTRUCTOR_DEFAULT // Default destructor
      );

      m_metal_library = m_device->newLibrary(metallib_dispatch_data, &error);
      if (!m_metal_library) {
        ICICLE_LOG_ERROR << "Failed to load Metal library: " << error->localizedDescription()->utf8String();
        ICICLE_ASSERT(false); // Fatal error, the library must be present
      }
    }

    // Destructor to release cached pipeline states and the Metal library
    ~MetalLibrary()
    {
      for (auto& pipeline_pair : m_pipeline_cache) {
        pipeline_pair.second->release(); // Release each cached pipeline state
      }
      if (m_metal_library) { m_metal_library->release(); }
    }

  private:
    MTL::Device* m_device;                                                        // Metal device for creating pipelines
    const std::string m_library_name;                                             // Name of the Metal library
    MTL::Library* m_metal_library = nullptr;                                      // Metal library handle
    std::unordered_map<std::string, MTL::ComputePipelineState*> m_pipeline_cache; // Cache of loaded kernels
    std::mutex m_mutex; // Mutex to protect access to the cache
  };

} // namespace icicle