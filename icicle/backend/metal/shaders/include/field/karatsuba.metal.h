#pragma once
#include "arithmetic.metal.h"

namespace karatsuba {

  static inline void mul_n(thread uint32_t* acc, thread const uint32_t* a, uint32_t bi, size_t n)
  {
    // TODO: convert to long4 and do vector mul?
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = arithmetic::mul_lo(a[i], bi);
      acc[i + 1] = arithmetic::mul_hi(a[i], bi);
    }
  }

  template <bool CARRY_IN = false>
  static inline void
  cmad_n(thread uint32_t* acc, thread const uint32_t* a, uint32_t bi, size_t n, thread uint32_t& carry)
  {
    acc[0] =
      CARRY_IN ? arithmetic::madc_lo_cc(a[0], bi, acc[0], carry) : arithmetic::mad_lo_cc(a[0], bi, acc[0], carry);
    acc[1] = arithmetic::madc_hi_cc(a[0], bi, acc[1], carry);

#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = arithmetic::madc_lo_cc(a[i], bi, acc[i], carry);
      acc[i + 1] = arithmetic::madc_hi_cc(a[i], bi, acc[i + 1], carry);
    }
  }

  template <bool EVEN_PHASE>
  static uint32_t cmad_n_msb(thread uint32_t* acc, thread const uint32_t* a, uint32_t bi, size_t n)
  {
    uint32_t carry = 0;
    if (EVEN_PHASE) {
      acc[0] = arithmetic::mad_lo_cc(a[0], bi, acc[0], carry);
      acc[1] = arithmetic::madc_hi_cc(a[0], bi, acc[1], carry);
    } else {
      acc[1] = arithmetic::mad_hi_cc(a[0], bi, acc[1], carry);
    }

#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = arithmetic::madc_lo_cc(a[i], bi, acc[i], carry);
      acc[i + 1] = arithmetic::madc_hi_cc(a[i], bi, acc[i + 1], carry);
    }
    return carry;
  }

  static inline void cmad_n_lsb(thread uint32_t* acc, thread const uint32_t* a, uint32_t bi, size_t n)
  {
    uint32_t carry = 0;
    if (n > 1)
      acc[0] = arithmetic::mad_lo_cc(a[0], bi, acc[0], carry);
    else
      acc[0] = arithmetic::mad_lo(a[0], bi, acc[0]);

    size_t i;
#pragma unroll
    for (i = 1; i < n - 1; i += 2) {
      acc[i] = arithmetic::madc_hi_cc(a[i - 1], bi, acc[i], carry);
      if (i == n - 2)
        acc[i + 1] = arithmetic::madc_lo(a[i + 1], bi, acc[i + 1], carry);
      else
        acc[i + 1] = arithmetic::madc_lo_cc(a[i + 1], bi, acc[i + 1], carry);
    }
    if (i == n - 1) acc[i] = arithmetic::madc_hi(a[i - 1], bi, acc[i], carry);
  }

  template <bool CARRY_OUT = false, bool CARRY_IN = false>
  uint32_t mad_row(
    thread uint32_t* odd,
    thread uint32_t* even,
    thread const uint32_t* a,
    uint32_t bi,
    size_t n,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    uint32_t carry = carry_for_low;
    cmad_n<CARRY_IN>(odd, a + 1, bi, n - 2, carry);
    odd[n - 2] = arithmetic::madc_lo_cc(a[n - 1], bi, ci, carry);
    odd[n - 1] =
      CARRY_OUT ? arithmetic::madc_hi_cc(a[n - 1], bi, di, carry) : arithmetic::madc_hi(a[n - 1], bi, di, carry);

    uint32_t cr = CARRY_OUT ? carry : 0;
    cmad_n(even, a, bi, n, carry);
    if (CARRY_OUT) {
      odd[n - 1] = arithmetic::addc_cc(odd[n - 1], carry_for_high, carry);
      cr = arithmetic::add(cr, carry);
    } else
      odd[n - 1] = arithmetic::addc(odd[n - 1], carry_for_high, carry);
    return cr;

    carry = 0;
    cmad_n(even, a, bi, n, carry);
    if (CARRY_OUT) {
      odd[n - 1] = arithmetic::addc_cc(odd[n - 1], carry_for_high, carry);
    } else
      odd[n - 1] = arithmetic::addc(odd[n - 1], carry_for_high, carry);
    return carry;
  }

  template <bool EVEN_PHASE>
  void mad_row_msb(thread uint32_t* odd, thread uint32_t* even, thread const uint32_t* a, uint32_t bi, size_t n)
  {
    uint32_t carry = cmad_n_msb<!EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = arithmetic::madc_lo_cc(a[n - 1], bi, 0u, carry);
    odd[EVEN_PHASE ? n : (n - 1)] = arithmetic::madc_hi(a[n - 1], bi, 0u, carry);
    carry = cmad_n_msb<EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = arithmetic::add(odd[EVEN_PHASE ? n : (n - 1)], carry);
  }

  static inline void
  mad_row_lsb(thread uint32_t* odd, thread uint32_t* even, thread const uint32_t* a, uint32_t bi, size_t n)
  {
    if (bi != 0) {
      if (n > 1) cmad_n_lsb(odd, a + 1, bi, n - 1);
      cmad_n_lsb(even, a, bi, n);
    }
    return;
  }

  static inline uint32_t
  mul_n_and_add(thread uint32_t* acc, thread const uint32_t* a, uint32_t bi, const thread uint32_t* extra, size_t n)
  {
    uint32_t carry = 0;
    acc[0] = arithmetic::mad_lo_cc(a[0], bi, extra[0], carry);

#pragma unroll
    for (size_t i = 1; i < n - 1; i += 2) {
      acc[i] = arithmetic::madc_hi_cc(a[i - 1], bi, extra[i], carry);
      acc[i + 1] = arithmetic::madc_lo_cc(a[i + 1], bi, extra[i + 1], carry);
    }

    acc[n - 1] = arithmetic::madc_hi_cc(a[n - 2], bi, extra[n - 1], carry);
    return carry;
  }

  /**
   * This method multiplies `a` and `b` (both assumed to have TLC / 2 limbs) and adds `in1` and `in2` (TLC limbs each)
   * to the result which is written to `even`.
   *
   * It is used to compute the "middle" part of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} =
   * (a_{hi} - a_{lo})(b_{lo} - b_{hi}) + a_{lo} \cdot b_{lo} + a_{hi} \cdot b_{hi} \f$. Currently this method assumes
   * that the top bit of \f$ a_{hi} \f$ and \f$ b_{hi} \f$ are unset. This ensures correctness by allowing to keep the
   * result inside TLC limbs and ignore the carries from the highest limb.
   */
  template <unsigned TLC>
  void multiply_and_add_short_raw(
    thread const uint32_t* a,
    thread const uint32_t* b,
    thread uint32_t* even,
    thread uint32_t* in1,
    thread uint32_t* in2)
  {
    uint32_t odd[TLC - 2];
    uint32_t first_row_carry = mul_n_and_add(even, a, b[0], in1, TLC >> 1);
    uint32_t carry = mul_n_and_add(odd, a + 1, b[0], &in2[1], TLC >> 1);

    size_t i;
#pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      carry = mad_row<true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], TLC >> 1, in1[(TLC >> 1) + i - 2], in1[(TLC >> 1) + i - 1], carry);
      carry =
        mad_row<true, false>(&odd[i], &even[i], a, b[i], TLC >> 1, in2[(TLC >> 1) + i - 1], in2[(TLC >> 1) + i], carry);
    }
    mad_row<false, true>(
      &even[TLC >> 1], &odd[(TLC >> 1) - 2], a, b[(TLC >> 1) - 1], TLC >> 1, in1[TLC - 2], in1[TLC - 1], carry,
      first_row_carry);

    carry = 0;
    // merge |even| and |odd| plus the parts of `in2` we haven't added yet (first and last limbs)
    even[0] = arithmetic::add_cc(even[0], in2[0], carry);
    for (i = 0; i < (TLC - 2); i++)
      even[i + 1] = arithmetic::addc_cc(even[i + 1], odd[i], carry);
    even[i + 1] = arithmetic::addc(even[i + 1], in2[i + 1], carry);
  }

  /**
   * This method multiplies `a` and `b` and writes the result into `even`. It assumes that `a` and `b` are TLC/2 limbs
   * long. The usual schoolbook algorithm is used.
   */
  template <unsigned TLC>
  static void multiply_short_raw(thread const uint32_t* a, thread const uint32_t* b, thread uint32_t* even)
  {
    uint32_t odd[TLC - 2];
    mul_n(even, a, b[0], TLC >> 1);
    mul_n(odd, a + 1, b[0], TLC >> 1);
    mad_row(&even[2], &odd[0], a, b[1], TLC >> 1);

    size_t i;
#pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      mad_row(&odd[i], &even[i], a, b[i], TLC >> 1);
      mad_row(&even[i + 2], &odd[i], a, b[i + 1], TLC >> 1);
    }

    uint32_t carry = 0;
    // merge |even| and |odd|
    even[1] = arithmetic::add_cc(even[1], odd[0], carry);
    for (i = 1; i < TLC - 2; i++)
      even[i + 1] = arithmetic::addc_cc(even[i + 1], odd[i], carry);
    even[i + 1] = arithmetic::add(even[i + 1], carry);
  }

  /**
   * This method multiplies `as` and `bs` and writes the (wide) result into `rs`.
   *
   * It is assumed that the highest bits of `as` and `bs` are unset which is true for all the numbers icicle had to deal
   * with so far. This method implements [subtractive
   * Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm#Implementation).
   */
  template <typename storage, typename wide_storage, unsigned TLC>
  void multiply_raw(const thread storage& as, const thread storage& bs, thread wide_storage& rs)
  {
    const thread uint32_t* a = as.limbs;
    const thread uint32_t* b = bs.limbs;
    thread uint32_t* r = rs.limbs;
    // TODO: Constexpr if is a c++17 feature
    if (TLC > 2) {
      // Next two lines multiply high and low halves of operands (\f$ a_{lo} \cdot b_{lo}; a_{hi} \cdot b_{hi} \$f) and
      // write the results into `r`.
      multiply_short_raw<TLC>(a, b, r);
      multiply_short_raw<TLC>(&a[TLC >> 1], &b[TLC >> 1], &r[TLC]);
      uint32_t middle_part[TLC];
      uint32_t diffs[TLC];
      // Differences of halves \f$ a_{hi} - a_{lo}; b_{lo} - b_{hi} \$f are written into `diffs`, signs written to
      // `carry1` and `carry2`.
      uint32_t carry1 = arithmetic::sub_u32<(TLC >> 1)>(&a[TLC >> 1], a, diffs);
      uint32_t carry2 = arithmetic::sub_u32<(TLC >> 1)>(b, &b[TLC >> 1], &diffs[TLC >> 1]);

      // Compute the "middle part" of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} \f$.
      // This is where the assumption about unset high bit of `a` and `b` is relevant.
      multiply_and_add_short_raw<TLC>(diffs, &diffs[TLC >> 1], middle_part, r, &r[TLC]);

      // Corrections that need to be performed when differences are negative.
      // Again, carry doesn't need to be propagated due to unset high bits of `a` and `b`.
      if (carry1) arithmetic::sub_u32<(TLC >> 1)>(&middle_part[TLC >> 1], &diffs[TLC >> 1], &middle_part[TLC >> 1]);
      if (carry2) arithmetic::sub_u32<(TLC >> 1)>(&middle_part[TLC >> 1], diffs, &middle_part[TLC >> 1]);
      // Now that middle part is fully correct, it can be added to the result.
      uint32_t carry = arithmetic::add_u32<TLC>(&r[TLC >> 1], middle_part, &r[TLC >> 1]);

      // Carry from adding middle part has to be propagated to the highest limb.
      for (size_t i = TLC + (TLC >> 1); i < 2 * TLC; i++)
        r[i] = arithmetic::add_cc(r[i], carry, carry);
    } else if (TLC == 2) {
      uint32_t odd[2];
      r[0] = arithmetic::mul_lo(a[0], b[0]);
      r[1] = arithmetic::mul_hi(a[0], b[0]);
      r[2] = arithmetic::mul_lo(a[1], b[1]);
      r[3] = arithmetic::mul_hi(a[1], b[1]);
      odd[0] = arithmetic::mul_lo(a[0], b[1]);
      odd[1] = arithmetic::mul_hi(a[0], b[1]);
      odd[0] = arithmetic::mad_lo(a[1], b[0], odd[0]);
      odd[1] = arithmetic::mad_hi(a[1], b[0], odd[1]);
      uint32_t carry;
      r[1] = arithmetic::add_cc(r[1], odd[0], carry);
      r[2] = arithmetic::addc_cc(r[2], odd[1], carry);
      r[3] = arithmetic::add(r[3], carry);
    } else if (TLC == 1) {
      // TODO: Here and in other places - consider using u64 mul instead of two u32 muls
      r[0] = arithmetic::mul_lo(a[0], b[0]);
      r[1] = arithmetic::mul_hi(a[0], b[0]);
    }
  }

  /**
   * A function that computes wide product \f$ rs = as \cdot bs \f$ that's correct for the higher TLC + 1 limbs with a
   * small maximum error.
   *
   * The way this function saves computations (as compared to regular school-book multiplication) is by not including
   * terms that are too small. Namely, limb product \f$ a_i \cdot b_j \f$ is excluded if \f$ i + j < TLC - 2 \f$ and
   * only the higher half is included if \f$ i + j = TLC - 2 \f$. All other limb products are included. So, the error
   * i.e. difference between true product and the result of this function written to `rs` is exactly the sum of all
   * dropped limbs products, which we can bound: \f$ a_0 \cdot b_0 + 2^{32}(a_0 \cdot b_1 + a_1 \cdot b_0) + \dots +
   * 2^{32(TLC - 3)}(a_{TLC - 3} \cdot b_0 + \dots + a_0 \cdot b_{TLC - 3}) + 2^{32(TLC - 2)}(\floor{\frac{a_{TLC - 2}
   * \cdot b_0}{2^{32}}} + \dots + \floor{\frac{a_0 \cdot b_{TLC - 2}}{2^{32}}}) \leq 2^{64} + 2\cdot 2^{96} + \dots +
   * (TLC - 2) \cdot 2^{32(TLC - 1)} + (TLC - 1) \cdot 2^{32(TLC - 1)} \leq 2(TLC - 1) \cdot 2^{32(TLC - 1)}\f$.
   */
  template <typename storage, typename wide_storage, unsigned TLC>
  static void multiply_msb_raw(const thread storage& as, const thread storage& bs, thread wide_storage& rs)
  {
    if constexpr (TLC > 1) {
      const thread uint32_t* a = as.limbs;
      const thread uint32_t* b = bs.limbs;
      thread uint32_t* even = rs.limbs;
      uint32_t odd[2 * TLC - 2];

      even[TLC - 1] = arithmetic::mul_hi(a[TLC - 2], b[0]);
      odd[TLC - 2] = arithmetic::mul_lo(a[TLC - 1], b[0]);
      odd[TLC - 1] = arithmetic::mul_hi(a[TLC - 1], b[0]);
      size_t i;
#pragma unroll
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row_msb<true>(&even[TLC - 2], &odd[TLC - 2], &a[TLC - i - 1], b[i - 1], i + 1);
        mad_row_msb<false>(&odd[TLC - 2], &even[TLC - 2], &a[TLC - i - 2], b[i], i + 2);
      }
      mad_row(&even[TLC], &odd[TLC - 2], a, b[TLC - 1], TLC);

      uint32_t carry = 0;
      // merge |even| and |odd|
      arithmetic::add_cc(even[TLC - 1], odd[TLC - 2], carry);
      for (i = TLC - 1; i < 2 * TLC - 2; i++)
        even[i + 1] = arithmetic::addc_cc(even[i + 1], odd[i], carry);
      even[i + 1] = arithmetic::add(even[i + 1], carry);
    } else {
      multiply_raw<storage, wide_storage, TLC>(as, bs, rs);
    }
  }

  /**
   * A function that computes the low half of the fused multiply-and-add \f$ rs = as \cdot bs + cs \f$ where
   * \f$ bs = 2^{32*nof_limbs} \f$.
   *
   * For efficiency, this method does not include terms that are too large. Namely, limb product \f$ a_i \cdot b_j \f$
   * is excluded if \f$ i + j > TLC - 1 \f$ and only the lower half is included if \f$ i + j = TLC - 1 \f$. All other
   * limb products are included.
   */
  template <typename storage, typename wide_storage, unsigned TLC>
  static void multiply_and_add_lsb_neg_modulus_raw(
    const thread storage& as, const thread storage& bs, const thread storage& cs, thread storage& rs)
  {
    const thread uint32_t* a = as.limbs;
    const thread uint32_t* b = bs.limbs;
    const thread uint32_t* c = cs.limbs;
    thread uint32_t* even = rs.limbs;

    if constexpr (TLC > 2) {
      uint32_t odd[TLC - 1];
      size_t i;
      // `b[0]` is \f$ 2^{32} \f$ minus the last limb of prime modulus. Because most scalar (and some base) primes
      // are necessarily NTT-friendly, `b[0]` often turns out to be \f$ 2^{32} - 1 \f$.
      if (b[0] == 0xFFFFFFFF) {
        arithmetic::sub_u32<TLC>(c, a, even);
        for (i = 0; i < TLC - 1; i++)
          odd[i] = a[i];
      } else {
        mul_n_and_add(even, a, b[0], c, TLC);
        mul_n(odd, a + 1, b[0], TLC - 1);
      }
      mad_row_lsb(&even[2], &odd[0], a, b[1], TLC - 1);

#pragma unroll
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row_lsb(&odd[i], &even[i], a, b[i], TLC - i);
        mad_row_lsb(&even[i + 2], &odd[i], a, b[i + 1], TLC - i - 1);
      }

      uint32_t carry = 0;
      // merge |even| and |odd|
      even[1] = arithmetic::add_cc(even[1], odd[0], carry);
      for (i = 1; i < TLC - 2; i++)
        even[i + 1] = arithmetic::addc_cc(even[i + 1], odd[i], carry);
      even[i + 1] = arithmetic::addc(even[i + 1], odd[i], carry);
    } else if (TLC == 2) {
      even[0] = arithmetic::mad_lo(a[0], b[0], c[0]);
      even[1] = arithmetic::mad_hi(a[0], b[0], c[0]);
      even[1] = arithmetic::mad_lo(a[0], b[1], even[1]);
      even[1] = arithmetic::mad_lo(a[1], b[0], even[1]);
    } else if (TLC == 1) {
      even[0] = a[0] * b[0] + c[0];
    }
  }
} // namespace karatsuba