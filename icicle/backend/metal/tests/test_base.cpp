#include "Metal/Metal.h"
#include <time.h>
#include <random>

#include <gtest/gtest.h>
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "metal_device_api.h"
#include "metal_library_loader.h"
#include "icicle/curves/params/bn254.h"
#include "icicle/vec_ops.h"

using FpMiliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf("%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());
#define END_TIMER_AVERAGE(timer, msg, enable, iters)                                                                   \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / iters);

class IcicleMetalTestBase : public ::testing::Test
{
public:
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();
    ICICLE_CHECK(icicle_set_device("METAL"));
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

template <typename T>
void metal_ec_add(const T* vec_a, const T* vec_b, T* vec_r, uint64_t size, const VecOpsConfig& config)
{
  size *= config.batch_size;

  // Note that Metal cannot work async when using host memory
  bool is_async = config.is_async;
  if (is_async) {
    if (!config.is_a_on_device || !config.is_b_on_device || !config.is_result_on_device) {
      ICICLE_LOG_WARNING << "Metal backend doesn't support async calls when using host memory.";
      is_async = false;
    }
  }

  // Handle Input-Output: if host memory, allocate and copy to device, else retrieve Metal buffers
  auto [bufferA, offsetA] = config.is_a_on_device ? metal::map_to_metal_buffer(vec_a)
                                                  : metal::allocate_and_copy_to_device(vec_a, size * sizeof(T));
  auto [bufferB, offsetB] = config.is_b_on_device ? metal::map_to_metal_buffer(vec_b)
                                                  : metal::allocate_and_copy_to_device(vec_b, size * sizeof(T));
  auto [bufferR, offsetR] = config.is_result_on_device ? metal::map_to_metal_buffer(vec_r)
                                                       : metal::allocate_and_copy_to_device(vec_r, size * sizeof(T));
  // Dispatch the kernel to the stream (=CommandQueue): default stream or config.stream
  MTL::Device* metal_device = metal::get_device();
  MTL::CommandQueue* cmdQueue = (config.stream != nullptr) ? reinterpret_cast<MTL::CommandQueue*>(config.stream)
                                                           : metal::get_default_command_queue();

  auto pipelineState = METAL_GET_PIPELINE("ec_add");

  MTL::CommandBuffer* commandBuffer = cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  // Bind buffers
  encoder->setBuffer(bufferA, offsetA, 0);
  encoder->setBuffer(bufferB, offsetB, 1);
  encoder->setBuffer(bufferR, offsetR, 2);
  // Bind constants
  encoder->setBytes(&size /*data*/, sizeof(size) /*length*/, 3 /*index*/);

  const int NOF_THREADS_PER_BLOCK = 256;
  const int NOF_BLOCKS = (size + NOF_THREADS_PER_BLOCK - 1) / NOF_THREADS_PER_BLOCK;
  MTL::Size gridSize = MTL::Size(NOF_BLOCKS, 1, 1);
  MTL::Size threadGroupSize = MTL::Size(NOF_THREADS_PER_BLOCK, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadGroupSize);
  encoder->endEncoding();

  if (is_async) {
    commandBuffer->commit();
    return;
  }

  // For synchronous case: wait for completion, copy output back to host and release allocated resources
  ICICLE_ASSERT(!is_async);
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  // commandBuffer->release(); // Seems that this is a double-free so not required

  if (commandBuffer->status() == MTL::CommandBufferStatusError) {
    std::cout << commandBuffer->error()->localizedDescription()->cString(NS::StringEncoding::ASCIIStringEncoding)
              << std::endl;
    std::cout << commandBuffer->error()->domain()->cString(NS::StringEncoding::ASCIIStringEncoding) << std::endl;
  }

  // Post compute: copy output and release allocated buffers
  if (!config.is_result_on_device) {
    std::memcpy(vec_r, bufferR->contents(), size * sizeof(T));
    bufferR->release();
  }

  if (!config.is_a_on_device) { bufferA->release(); }
  if (!config.is_b_on_device) { bufferB->release(); }

  return;
}

template <typename T>
void random_samples(T* arr, uint64_t count)
{
  for (uint64_t i = 0; i < count; i++)
    arr[i] = i < 1000 ? T::rand_host() : arr[i - 1000];
}

using namespace icicle;
typedef bn254::projective_t TypeParam;

TEST_F(IcicleMetalTestBase, ProjectiveAdd)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << 8;
  const int batch_size = 1;
  const bool columns_batch = 1;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size = N * batch_size;

  auto config = default_vec_ops_config();
  config.batch_size = batch_size;
  config.columns_batch = columns_batch;

  TypeParam *in_a, *in_b, *out_main, *out_ref;
  ICICLE_CHECK(icicle_malloc((void**)&in_a, total_size * sizeof(TypeParam)));
  ICICLE_CHECK(icicle_malloc((void**)&in_b, total_size * sizeof(TypeParam)));
  ICICLE_CHECK(icicle_malloc((void**)&out_main, total_size * sizeof(TypeParam)));
  ICICLE_CHECK(icicle_malloc((void**)&out_ref, total_size * sizeof(TypeParam)));

  random_samples(in_a, total_size);
  random_samples(in_b, total_size);

  metal_ec_add(in_a, in_b, out_main, N, config);
  START_TIMER(EC_ADD)
  metal_ec_add(in_a, in_b, out_main, N, config);
  END_TIMER(EC_ADD, "Metal ec add", true);

  START_TIMER(EC_ADD_CPU)
  for (int i = 0; i < total_size; i++) {
    out_ref[i] = in_a[i] + in_b[i];
  }
  END_TIMER(EC_ADD_CPU, "CPU ec add", true);

  // for (int i = 0; i < total_size; i++) {
  //   std::cout << "Main: " << out_main[i] << "; Ref: " << out_ref[i] << std::endl;
  // }
  ASSERT_EQ(0, memcmp(out_main, out_ref, total_size * sizeof(TypeParam)));
}