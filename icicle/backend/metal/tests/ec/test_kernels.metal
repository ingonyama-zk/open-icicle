#include "curve/params/bn254.metal.h"

[[host_name("test_ec_add")]] kernel void ec_add(
  device const bn254ProjectiveG1* a [[buffer(0)]],
  device const bn254ProjectiveG1* b [[buffer(1)]],
  device bn254ProjectiveG1* result [[buffer(2)]],
  constant uint64_t& size [[buffer(3)]],
  uint BlockDim [[threads_per_threadgroup]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint ThreadIdx [[thread_position_in_threadgroup]])
{
  uint tid = blockIdx * BlockDim + ThreadIdx;
  if (tid >= size) { return; }
  result[tid] = a[tid] + b[tid];
}
