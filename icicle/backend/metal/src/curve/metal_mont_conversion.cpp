#include "Metal/Metal.h"

#include "icicle/curves/montgomery_conversion.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/curves/curve_config.h"
#include "mont_conversion.h"

using namespace curve_config;
using namespace icicle;

constexpr char affine_convert_montgomery_kernel_name[] = "affine_convert_montgomery";
REGISTER_AFFINE_CONVERT_MONTGOMERY_BACKEND(
  "METAL", (metal_convert_montgomery<affine_t, affine_convert_montgomery_kernel_name>));
constexpr char projective_convert_montgomery_kernel_name[] = "projective_convert_montgomery";
REGISTER_PROJECTIVE_CONVERT_MONTGOMERY_BACKEND(
  "METAL", (metal_convert_montgomery<projective_t, projective_convert_montgomery_kernel_name>));

#ifdef G2_ENABLED
constexpr char affine_g2_convert_montgomery_kernel_name[] = "affine_g2_convert_montgomery";
REGISTER_AFFINE_G2_CONVERT_MONTGOMERY_BACKEND(
  "METAL", (metal_convert_montgomery<g2_affine_t, affine_g2_convert_montgomery_kernel_name>));
constexpr char projective_g2_convert_montgomery_kernel_name[] = "projective_g2_convert_montgomery";
REGISTER_PROJECTIVE_G2_CONVERT_MONTGOMERY_BACKEND(
  "METAL", (metal_convert_montgomery<g2_projective_t, projective_g2_convert_montgomery_kernel_name>));
#endif // G2