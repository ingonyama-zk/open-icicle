#include "metal_msm_backend.h"

using namespace curve_config;
using namespace icicle;

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("METAL", (metal_msm_precompute_bases<scalar_t, affine_t, projective_t>));
REGISTER_MSM_BACKEND("METAL", (msm_metal<scalar_t, affine_t, projective_t>));