#pragma once

#include "icicle/fields/id.h"

// Compiled shaders (.metallib) are embedded in each icicle-metal library.
// To load and dispatch kernels, we need to know the symbol of this embedded data.
// Each backend library must define a unique identifier, such as 'METAL_LIBRARY_ID_CURVE_BN254' or similar
// so that exactly one case is defined per metal library. This case defines names and types that allow
// 'metal_library_loader.h' to retrieve the compiled shaders

#ifdef METAL_LIBRARY_ID_CURVE_BN254
extern unsigned char curve_bn254_metallib[];
extern unsigned int curve_bn254_metallib_len;
struct METAL_LIB_TYPE_CURVE_BN254 {
  static inline constexpr auto& metallib_data = curve_bn254_metallib;
  static inline constexpr auto& metallib_data_len = curve_bn254_metallib_len;
  static inline constexpr const char* metallib_library_prefix = "curve_bn254_";
};
using METAL_LIB = METAL_LIB_TYPE_CURVE_BN254;

#elif defined(METAL_LIBRARY_ID_CURVE_BLS12_381)
extern unsigned char curve_bls12_381_metallib[];
extern unsigned int curve_bls12_381_metallib_len;
struct METAL_LIB_TYPE_CURVE_BLS12_381 {
  static inline constexpr auto& metallib_data = curve_bls12_381_metallib;
  static inline constexpr auto& metallib_data_len = curve_bls12_381_metallib_len;
  static inline constexpr const char* metallib_library_prefix = "curve_bls12_381_";
};
using METAL_LIB = METAL_LIB_TYPE_CURVE_BLS12_381;

#elif defined(METAL_LIBRARY_ID_FIELD_BN254)
extern unsigned char field_bn254_metallib[];
extern unsigned int field_bn254_metallib_len;
struct METAL_LIB_TYPE_FIELD_BN254 {
  static inline constexpr auto& metallib_data = field_bn254_metallib;
  static inline constexpr auto& metallib_data_len = field_bn254_metallib_len;
  static inline constexpr const char* metallib_library_prefix = "field_bn254_";
};
using METAL_LIB = METAL_LIB_TYPE_FIELD_BN254;

#elif defined(METAL_LIBRARY_ID_FIELD_BLS12_381)
extern unsigned char field_bls12_381_metallib[];
extern unsigned int field_bls12_381_metallib_len;
struct METAL_LIB_TYPE_FIELD_BLS12_381 {
  static inline constexpr auto& metallib_data = field_bls12_381_metallib;
  static inline constexpr auto& metallib_data_len = field_bls12_381_metallib_len;
  static inline constexpr const char* metallib_library_prefix = "field_bls12_381_";
};
using METAL_LIB = METAL_LIB_TYPE_FIELD_BLS12_381;

#elif defined(METAL_LIBRARY_ID_TEST)
extern unsigned char test_metallib[];
extern unsigned int test_metallib_len;
struct METAL_LIB_TYPE_TEST {
  static inline constexpr auto& metallib_data = test_metallib;
  static inline constexpr auto& metallib_data_len = test_metallib_len;
  static inline constexpr const char* metallib_library_prefix = "test_";
};
using METAL_LIB = METAL_LIB_TYPE_TEST;
#else
  #error "Unsupported ICICLE library specified."
#endif
