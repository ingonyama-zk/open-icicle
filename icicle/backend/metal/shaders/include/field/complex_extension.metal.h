#pragma once

#include "field/field.metal.h"

template <typename CONFIG, typename BaseField>
class ComplexExtensionField
{
  typedef typename BaseField::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide imaginary;

    constexpr ComplexExtensionField reduce() const
    {
      return ComplexExtensionField{this->real.reduce(), this->imaginary.reduce()};
    }

    FWide operator+(const thread FWide& y) const device
    {
      const FWide x = *this;
      return x + y;
    }

    FWide operator+(const device FWide& other) const device
    {
      const FWide a = *this;
      const FWide b = other;
      return a + b;
    }

    FWide operator+(thread const FWide& ys) const
    {
      return ExtensionWide{this->real + ys.real, this->imaginary + ys.imaginary};
    }

    FWide operator-(const device FWide& other) const device
    {
      const FWide a = *this;
      const FWide b = other;
      return a - b;
    }

    FWide operator-(const thread FWide& y) const device
    {
      const FWide x = *this;
      return x - y;
    }

    FWide operator-(thread const FWide& ys) const
    {
      return ExtensionWide{this->real - ys.real, this->imaginary - ys.imaginary};
    }
  };

public:
  BaseField real;
  BaseField imaginary;
  static constexpr ComplexExtensionField zero() { return ComplexExtensionField{BaseField::zero(), BaseField::zero()}; }

  static constexpr ComplexExtensionField one() { return ComplexExtensionField{BaseField::one(), BaseField::zero()}; }
  bool is_zero() const { return this->real.is_zero() && this->imaginary.is_zero(); }

  ComplexExtensionField operator+(const device ComplexExtensionField& other) const device
  {
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return a + b;
  }

  ComplexExtensionField operator+(const thread ComplexExtensionField& other) const device
  {
    const ComplexExtensionField x = *this;
    return x + other;
  }

  ComplexExtensionField operator+(const thread ComplexExtensionField& other) const thread
  {
    return ComplexExtensionField{this->real + other.real, this->imaginary + other.imaginary};
  }

  ComplexExtensionField operator+(const device BaseField& other) const device
  {
    const ComplexExtensionField a = *this;
    const BaseField b = other;
    return a + b;
  }

  ComplexExtensionField operator+(const thread BaseField& other) const device
  {
    const ComplexExtensionField x = *this;
    return x + other;
  }

  ComplexExtensionField operator+(const thread BaseField& other) const thread
  {
    return ComplexExtensionField{this->real + other, this->imaginary};
  }

  friend ComplexExtensionField operator+(const device BaseField& xs, const device ComplexExtensionField& ys)
  {
    const BaseField a = xs;
    const ComplexExtensionField b = ys;
    return a + b;
  }

  friend ComplexExtensionField operator+(const thread BaseField& xs, const thread ComplexExtensionField& ys)
  {
    return ComplexExtensionField{ys.real + xs, ys.imaginary};
  }

  ComplexExtensionField operator/(const device ComplexExtensionField& other) const device
  {
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return a / b;
  }

  ComplexExtensionField operator/(const thread ComplexExtensionField& denominator) const thread
  {
    ComplexExtensionField inv = inverse(denominator);
    return *this * inv;
  }

  bool operator==(const device ComplexExtensionField& other) const device
  {
    // move from device to thread memory
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return a == b;
  }

  bool operator==(const thread ComplexExtensionField& other) const thread
  {
    return this->real == other.real && this->imaginary == other.imaginary;
  }

  bool operator!=(const device ComplexExtensionField& other) const device
  {
    // move from device to thread memory
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return !(a == b);
  }

  bool operator!=(const thread ComplexExtensionField& other) const thread { return !(*this == other); }

  ComplexExtensionField operator-(const device ComplexExtensionField& other) const device
  {
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return a - b;
  }

  ComplexExtensionField operator-(const thread ComplexExtensionField& y) const device
  {
    const ComplexExtensionField x = *this;
    return x - y;
  }

  ComplexExtensionField operator-(const thread ComplexExtensionField& other) const thread
  {
    return ComplexExtensionField{this->real - other.real, this->imaginary - other.imaginary};
  }

  ComplexExtensionField operator*(const device ComplexExtensionField& other) const device
  {
    const ComplexExtensionField a = *this;
    const ComplexExtensionField b = other;
    return a * b;
  }

  ComplexExtensionField operator*(const thread ComplexExtensionField& scalar) const device
  {
    const ComplexExtensionField element = *this;
    return scalar * element;
  }

  ComplexExtensionField operator*(const thread ComplexExtensionField& other) const thread
  {
    ExtensionWide xy = mul_wide(*this, other); // full mult
    return xy.reduce();                        // reduce mod p
  }

  ComplexExtensionField pow(uint32_t exp) const device
  {
    ComplexExtensionField base = *this;
    ComplexExtensionField res = ComplexExtensionField::one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }

  constexpr ComplexExtensionField sqr() const
  {
    // TODO: change to a more efficient squaring
    return *this * *this;
  }

  constexpr ExtensionWide sqr_wide() const
  {
    // TODO: change to a more efficient squaring
    return mul_wide(*this, *this);
  }

  template <thread const ComplexExtensionField& M>
  constexpr ComplexExtensionField mul_const() const
  {
    constexpr BaseField mul_real = M.real;
    constexpr BaseField mul_imaginary = M.imaginary;
    const BaseField xs_real = this->real;
    const BaseField xs_imaginary = this->imaginary;
    BaseField real_prod = xs_real.template mul_const<mul_real>();
    BaseField imaginary_prod = xs_imaginary.template mul_const<mul_imaginary>();
    BaseField re_im = xs_imaginary.template mul_const<mul_real>();
    BaseField im_re = xs_real.template mul_const<mul_imaginary>();
    BaseField nonresidue_times_im = imaginary_prod.template mul_unsigned<CONFIG::nonresidue>();
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    return ComplexExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t M>
  constexpr ComplexExtensionField mul_unsigned() const
  {
    return ComplexExtensionField{this->real.template mul_unsigned<M>(), this->imaginary.template mul_unsigned<M>()};
  }

  constexpr ComplexExtensionField neg() const { return ComplexExtensionField{this->real.neg(), this->imaginary.neg()}; }

  static ComplexExtensionField inverse(const device ComplexExtensionField& xs)
  {
    ComplexExtensionField x = xs;
    return inverse(x);
  }

  static ComplexExtensionField inverse(const thread ComplexExtensionField& xs)
  {
    ComplexExtensionField xs_conjugate = {xs.real, xs.imaginary.neg()};
    BaseField nonresidue_times_im = xs.imaginary.sqr().template mul_unsigned<CONFIG::nonresidue>();
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    // TODO: wide here
    BaseField xs_norm_squared = xs.real.sqr() - nonresidue_times_im;
    return xs_conjugate * ComplexExtensionField{BaseField::inverse(xs_norm_squared), BaseField::zero()};
  }

  static constexpr ComplexExtensionField to_montgomery(const device ComplexExtensionField& value)
  {
    const ComplexExtensionField x = value;
    return to_montgomery(x);
  }

  static constexpr ComplexExtensionField to_montgomery(const thread ComplexExtensionField& value)
  {
    return ComplexExtensionField{BaseField::to_montgomery(value.real), BaseField::to_montgomery(value.imaginary)};
  }

  static constexpr ComplexExtensionField from_montgomery(const device ComplexExtensionField& value)
  {
    const ComplexExtensionField x = value;
    return from_montgomery(x);
  }

  static constexpr ComplexExtensionField from_montgomery(const thread ComplexExtensionField& value)
  {
    return ComplexExtensionField{BaseField::from_montgomery(value.real), BaseField::from_montgomery(value.imaginary)};
  }

  static constexpr ExtensionWide
  mul_wide(const thread ComplexExtensionField& xs, const thread ComplexExtensionField& ys)
  {
    FWide real_prod = BaseField::mul_wide(xs.real, ys.real);
    FWide imaginary_prod = BaseField::mul_wide(xs.imaginary, ys.imaginary);
    FWide prod_of_sums = BaseField::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
    FWide nonresidue_times_im = imaginary_prod.template mul_unsigned<CONFIG::nonresidue>();
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    return ExtensionWide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  static constexpr ExtensionWide mul_wide(const thread ComplexExtensionField& xs, const thread BaseField& ys)
  {
    return ExtensionWide{BaseField::mul_wide(xs.real, ys), BaseField::mul_wide(xs.imaginary, ys)};
  }
};
