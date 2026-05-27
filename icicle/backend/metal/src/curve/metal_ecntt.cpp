#include "icicle/backend/ecntt_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename E>
eIcicleError
metal_ntt(const Device& device, const E* input, int size, NTTDir dir, const NTTConfig<S>& config, E* output)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_ECNTT_BACKEND("METAL", (metal_ntt<scalar_t, projective_t>));