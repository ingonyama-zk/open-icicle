#pragma once
#include <metal_integer>

namespace arithmetic {
  // return x + y with T operands
  template <typename T>
  static constexpr T add(const T x, const T y)
  {
    return x + y;
  }

  // return x + y + carry with T operands
  template <typename T>
  static inline constexpr T addc(const T x, const T y, const T carry)
  {
    return x + y + carry;
  }

  // return x + y and carry out with T operands
  template <typename T>
  static constexpr T add_cc(const T x, const T y, thread T& carry)
  {
    T result = x + y;
    carry = x > result;
    return result;
  }

  // return x + y + carry and carry out  with T operands
  template <typename T>
  static constexpr T addc_cc(const T x, const T y, thread T& carry)
  {
    const T result = x + y + carry;
    carry = carry && x >= result || !carry && x > result;
    return result;
  }

  // return x - y with T operands
  template <typename T>
  static constexpr T sub(const T x, const T y)
  {
    return x - y;
  }

  //    return x - y - borrow with T operands
  template <typename T>
  static constexpr T subc(const T x, const T y, const T borrow)
  {
    return x - y - borrow;
  }

  //    return x - y and borrow out with T operands
  template <typename T>
  static constexpr T sub_cc(const T x, const T y, thread T& borrow)
  {
    T result = x - y;
    borrow = x < result;
    return result;
  }

  //    return x - y - borrow and borrow out with T operands
  template <typename T>
  static constexpr T subc_cc(const T x, const T y, thread T& borrow)
  {
    const T result = x - y - borrow;
    borrow = borrow && x <= result || !borrow && x < result;
    return result;
  }

  //    return low 32 bits of x * y
  template <typename T>
  static constexpr T mul_lo(const T x, const T y)
  {
    return x * y;
  }

  //    return high 32 bits of x * y
  template <typename T>
  static constexpr T mul_hi(const T x, const T y)
  {
    return metal::mulhi(x, y);
  }

  //    return (low 32 bits of x * y) + z
  template <typename T>
  static constexpr T mad_lo(const T x, const T y, const T z)
  {
    return add(x * y, z);
  }

  //    return (low 32 bits of x * y) + z + carry
  template <typename T>
  static constexpr T madc_lo(const T x, const T y, const T z, thread const T& carry)
  {
    return addc(x * y, z, carry);
  }

  //    return (low 32 bits of x * y) + z and carry out
  template <typename T>
  static constexpr T mad_lo_cc(const T x, const T y, const T z, thread T& carry)
  {
    return add_cc(x * y, z, carry);
  }

  //    return (low 32 bits of x * y) + z + carry and carry out
  template <typename T>
  static constexpr T madc_lo_cc(const T x, const T y, const T z, thread T& carry)
  {
    return addc_cc(x * y, z, carry);
  }

  //    return (high 32 bits of x * y) + z and carry out
  template <typename T>
  static constexpr T mad_hi_cc(const T x, const T y, const T z, thread T& carry)
  {
    return add_cc(metal::mulhi(x, y), z, carry);
  }

  //    return (high 32 bits of x * y) + z
  template <typename T>
  static constexpr T mad_hi(const T x, const T y, const T z)
  {
    return add(metal::mulhi(x, y), z);
  }

  //    return (high 32 bits of x * y) + z + carry
  template <typename T>
  static constexpr T madc_hi(const T x, const T y, const T z, thread const T& carry)
  {
    return addc(metal::mulhi(x, y), z, carry);
  }

  //    return (high 32 bits of x * y) + z + carry and carry out
  template <typename T>
  static constexpr T madc_hi_cc(const T x, const T y, const T z, thread T& carry)
  {
    return addc_cc(metal::mulhi(x, y), z, carry);
  }

  // return x * y + z + carry and carry out with uint32_t operands
  static constexpr uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, thread uint32_t& carry)
  {
    uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
    carry = (uint32_t)(r >> 32);
    uint32_t result = r & 0xffffffff;
    return result;
  }

  static constexpr uint64_t madc_cc_64(const uint64_t x, const uint64_t y, const uint64_t z, thread uint64_t& carry)
  {
    __uint128_t r = static_cast<__uint128_t>(x) * y + z + carry;
    carry = (uint64_t)(r >> 64);
    uint64_t result = r & 0xffffffffffffffff;
    return result;
  }

  template <unsigned NLIMBS>
  static constexpr uint32_t add_u32(thread const uint32_t* x, thread const uint32_t* y, thread uint32_t* r)
  {
    uint32_t carry = 0;
    r[0] = arithmetic::add_cc(x[0], y[0], carry);
    for (unsigned i = 1; i < NLIMBS; i++)
      r[i] = arithmetic::addc_cc(x[i], y[i], carry);
    return carry;
  }

  template <unsigned NLIMBS>
  static constexpr uint32_t sub_u32(thread const uint32_t* x, thread const uint32_t* y, thread uint32_t* r)
  {
    uint32_t borrow = 0;
    r[0] = arithmetic::sub_cc(x[0], y[0], borrow);
    for (unsigned i = 1; i < NLIMBS; i++)
      r[i] = arithmetic::subc_cc(x[i], y[i], borrow);
    return borrow;
  }
} // namespace arithmetic