#pragma once

#include "curve/projective.metal.h"
#include "field/params/bn254.metal.h"

// Instantiate Projective type
struct G1;
// typedef Affine<point_field_t> affine_t;

using bn254AffineG1 = Affine<bn254BaseField>;
using bn254ProjectiveG1 = Projective<bn254BaseField, bn254Scalar, G1>;
// using bn254ProjectiveG2 = Projective<bn254G2BaseField, bn254Scalar, G2>;

struct G1 {
  static constant constexpr bn254BaseField gen_x = {0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
  static constant constexpr bn254BaseField gen_y = {0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
  static constant constexpr bn254BaseField weierstrass_b = {0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

  static constant constexpr bool is_b_u32 = true;
  static constant constexpr bool is_b_neg = false;
  static constant constexpr uint32_t mu = 3834012553;
}; // G1