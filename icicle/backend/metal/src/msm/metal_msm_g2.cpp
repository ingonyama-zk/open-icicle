#include "metal_msm_backend.h"
#include "icicle/runtime.h"
#include "icicle/device.h"
#include "icicle/msm.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename A, typename P>
eIcicleError msm_metal_cpu_mock(
  const Device& device, const S* scalars, const A* points, int msm_size, const MSMConfig& config, P* results)
{
  icicle_set_device({"CPU", 0});
  auto result = msm<S, A, P>(scalars, points, msm_size, config, results);
  icicle_set_device({"METAL", 0});
  return result;
}

template <typename S, typename A, typename P>
eIcicleError metal_msm_precompute_bases_cpu_mock(
  const Device& device, const A* input_bases, int nof_bases, const MSMConfig& config, A* output_bases)
{
  icicle_set_device({"CPU", 0});
  auto result = msm_precompute_bases<A>(input_bases, nof_bases, config, output_bases);
  icicle_set_device({"METAL", 0});
  return result;
}

REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(
  "METAL", (metal_msm_precompute_bases_cpu_mock<scalar_t, g2_affine_t, g2_projective_t>));
REGISTER_MSM_G2_BACKEND("METAL", (msm_metal_cpu_mock<scalar_t, g2_affine_t, g2_projective_t>));