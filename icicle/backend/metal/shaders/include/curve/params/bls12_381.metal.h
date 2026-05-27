#pragma once

#include "curve/projective.metal.h"
#include "field/params/bls12_381.metal.h"

// Instantiate Projective type
struct G1;
// typedef Affine<point_field_t> affine_t;

using bls12_381AffineG1 = Affine<bls12_381BaseField>;
using bls12_381ProjectiveG1 = Projective<bls12_381BaseField, bls12_381Scalar, G1>;
// using bls12_381ProjectiveG2 = Projective<bls12_381G2BaseField, bls12_381Scalar, G2>;

struct G1 {
  static constant constexpr bls12_381BaseField gen_x = {0xdb22c6bb, 0xfb3af00a, 0xf97a1aef, 0x6c55e83f,
                                                        0x171bac58, 0xa14e3a3f, 0x9774b905, 0xc3688c4f,
                                                        0x4fa9ac0f, 0x2695638c, 0x3197d794, 0x17f1d3a7};
  static constant constexpr bls12_381BaseField gen_y = {0x46c5e7e1, 0xcaa2329,  0xa2888ae4, 0xd03cc744,
                                                        0x2c04b3ed, 0xdb18cb,   0xd5d00af6, 0xfcf5e095,
                                                        0x741d8ae4, 0xa09e30ed, 0xe3aaa0f1, 0x8b3f481};
  static constant constexpr bls12_381BaseField weierstrass_b = {0x4, 0x0, 0x0, 0x0, 0x0, 0x0,
                                                                0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

  static constant constexpr bool is_b_u32 = true;
  static constant constexpr bool is_b_neg = false;
  static constant constexpr uint32_t mu = 4294770685;
}; // G1