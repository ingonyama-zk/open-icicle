#pragma once

#include "field/field.metal.h"

template <typename CONFIG, typename BaseField>
class QuarticExtensionField
{
  typedef typename BaseField::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide im1;
    FWide im2;
    FWide im3;

    constexpr QuarticExtensionField reduce() const
    {
      return QuarticExtensionField{this->real.reduce(), this->im1.reduce(), this->im2.reduce(), this->im3.reduce()};
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
      return ExtensionWide{this->real + ys.real, this->im1 + ys.im1, this->im2 + ys.im2, this->im3 + ys.im3};
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
      return ExtensionWide{this->real - ys.real, this->im1 - ys.im1, this->im2 - ys.im2, this->im3 - ys.im3};
    }
  };

public:
  BaseField real;
  BaseField im1;
  BaseField im2;
  BaseField im3;
  static constexpr QuarticExtensionField zero()
  {
    return QuarticExtensionField{BaseField::zero(), BaseField::zero(), BaseField::zero(), BaseField::zero()};
  }

  static constexpr QuarticExtensionField one()
  {
    return QuarticExtensionField{BaseField::one(), BaseField::zero(), BaseField::zero(), BaseField::zero()};
  }
  bool is_zero() const
  {
    return this->real.is_zero() && this->im1.is_zero() && this->im2.is_zero() && this->im3.is_zero();
  }

  QuarticExtensionField operator+(const device QuarticExtensionField& other) const device
  {
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return a + b;
  }

  QuarticExtensionField operator+(const thread QuarticExtensionField& other) const device
  {
    const QuarticExtensionField x = *this;
    return x + other;
  }

  QuarticExtensionField operator+(const thread QuarticExtensionField& other) const thread
  {
    return QuarticExtensionField{
      this->real + other.real, this->im1 + other.im1, this->im2 + other.im2, this->im3 + other.im3};
  }

  QuarticExtensionField operator+(const device BaseField& other) const device
  {
    const QuarticExtensionField a = *this;
    const BaseField b = other;
    return a + b;
  }

  QuarticExtensionField operator+(const thread BaseField& other) const device
  {
    const QuarticExtensionField x = *this;
    return x + other;
  }

  QuarticExtensionField operator+(const thread BaseField& other) const thread
  {
    return QuarticExtensionField{this->real + other, this->im1, this->im2, this->im3};
  }

  friend QuarticExtensionField operator+(const device BaseField& xs, const device QuarticExtensionField& ys)
  {
    const BaseField a = xs;
    const QuarticExtensionField b = ys;
    return a + b;
  }

  friend QuarticExtensionField operator+(const thread BaseField& xs, const thread QuarticExtensionField& ys)
  {
    return QuarticExtensionField{ys.real + xs, ys.im1, ys.im2, ys.im3};
  }

  QuarticExtensionField operator/(const device QuarticExtensionField& other) const device
  {
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return a / b;
  }

  QuarticExtensionField operator/(const thread QuarticExtensionField& denominator) const thread
  {
    QuarticExtensionField inv = inverse(denominator);
    return *this * inv;
  }

  bool operator==(const device QuarticExtensionField& other) const device
  {
    // move from device to thread memory
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return a == b;
  }

  bool operator==(const thread QuarticExtensionField& other) const thread
  {
    return this->real == other.real && this->im1 == other.im1 && this->im2 == other.im2 && this->im3 == other.im3;
  }

  bool operator!=(const device QuarticExtensionField& other) const device
  {
    // move from device to thread memory
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return !(a == b);
  }

  bool operator!=(const thread QuarticExtensionField& other) const thread { return !(*this == other); }

  QuarticExtensionField operator-(const device QuarticExtensionField& other) const device
  {
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return a - b;
  }

  QuarticExtensionField operator-(const thread QuarticExtensionField& y) const device
  {
    const QuarticExtensionField x = *this;
    return x - y;
  }

  QuarticExtensionField operator-(const thread QuarticExtensionField& other) const thread
  {
    return QuarticExtensionField{
      this->real - other.real, this->im1 - other.im1, this->im2 - other.im2, this->im3 - other.im3};
  }

  QuarticExtensionField operator*(const device QuarticExtensionField& other) const device
  {
    const QuarticExtensionField a = *this;
    const QuarticExtensionField b = other;
    return a * b;
  }

  QuarticExtensionField operator*(const thread QuarticExtensionField& scalar) const device
  {
    const QuarticExtensionField element = *this;
    return scalar * element;
  }

  QuarticExtensionField operator*(const thread QuarticExtensionField& other) const thread
  {
    ExtensionWide xy = mul_wide(*this, other); // full mult
    return xy.reduce();                        // reduce mod p
  }

  QuarticExtensionField pow(uint32_t exp) const device
  {
    QuarticExtensionField base = *this;
    QuarticExtensionField res = QuarticExtensionField::one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }

  constexpr QuarticExtensionField sqr() const
  {
    // TODO: change to a more eBaseFieldicient squaring
    return *this * *this;
  }

  constexpr ExtensionWide sqr_wide() const
  {
    // TODO: change to a more eBaseFieldicient squaring
    return mul_wide(*this, *this);
  }

  template <uint32_t M>
  constexpr QuarticExtensionField mul_unsigned() const
  {
    return QuarticExtensionField{
      this->real.template mul_unsigned<M>(), this->im1.template mul_unsigned<M>(), this->im2.template mul_unsigned<M>(),
      this->im3.template mul_unsigned<M>()};
  }

  constexpr QuarticExtensionField neg() const
  {
    return QuarticExtensionField{this->real.neg(), this->im1.neg(), this->im2.neg(), this->im3.neg()};
  }

  static QuarticExtensionField inverse(const device QuarticExtensionField& xs)
  {
    QuarticExtensionField x = xs;
    return inverse(x);
  }

  static QuarticExtensionField inverse(const thread QuarticExtensionField& xs)
  {
    BaseField x, x0, x2;
    if (CONFIG::nonresidue_is_negative) {
      x0 =
        (xs.real.sqr_wide() +
         (BaseField::mul_wide(xs.im1, xs.im3 + xs.im3) - xs.im2.sqr_wide()).template mul_unsigned<CONFIG::nonresidue>())
          .reduce();
      x2 = (BaseField::mul_wide(xs.real, xs.im2 + xs.im2) - xs.im1.sqr_wide() +
            xs.im3.sqr_wide().template mul_unsigned<CONFIG::nonresidue>())
             .reduce();
      x = (x0.sqr_wide() + x2.sqr_wide().template mul_unsigned<CONFIG::nonresidue>()).reduce();
    } else {
      x0 =
        (xs.real.sqr_wide() -
         (BaseField::mul_wide(xs.im1, xs.im3 + xs.im3) - xs.im2.sqr_wide()).template mul_unsigned<CONFIG::nonresidue>())
          .reduce();
      x2 = (BaseField::mul_wide(xs.real, xs.im2 + xs.im2) - xs.im1.sqr_wide() -
            xs.im3.sqr_wide().template mul_unsigned<CONFIG::nonresidue>())
             .reduce();
      x = (x0.sqr_wide() - x2.sqr_wide().template mul_unsigned<CONFIG::nonresidue>()).reduce();
    }
    BaseField x_inv = BaseField::inverse(x);
    x0 = x0 * x_inv;
    x2 = x2 * x_inv;
    return {
      (CONFIG::nonresidue_is_negative ? (BaseField::mul_wide(xs.real, x0) +
                                         BaseField::mul_wide(xs.im2, x2).template mul_unsigned<CONFIG::nonresidue>())
                                      : (BaseField::mul_wide(xs.real, x0) -
                                         BaseField::mul_wide(xs.im2, x2).template mul_unsigned<CONFIG::nonresidue>()))
        .reduce(),
      ((CONFIG::nonresidue_is_negative
          ? (BaseField::mul_wide(xs.im3, x2).template mul_unsigned<CONFIG::nonresidue>()).neg()
          : BaseField::mul_wide(xs.im3, x2).template mul_unsigned<CONFIG::nonresidue>()) -
       BaseField::mul_wide(xs.im1, x0))
        .reduce(),
      (BaseField::mul_wide(xs.im2, x0) - BaseField::mul_wide(xs.real, x2)).reduce(),
      (BaseField::mul_wide(xs.im1, x2) - BaseField::mul_wide(xs.im3, x0)).reduce(),
    };
  }

  static constexpr QuarticExtensionField to_montgomery(const device QuarticExtensionField& value)
  {
    const QuarticExtensionField x = value;
    return to_montgomery(x);
  }

  static constexpr QuarticExtensionField to_montgomery(const thread QuarticExtensionField& value)
  {
    return QuarticExtensionField{
      BaseField::to_montgomery(value.real), BaseField::to_montgomery(value.im1), BaseField::to_montgomery(value.im2),
      BaseField::to_montgomery(value.im3)};
  }

  static constexpr QuarticExtensionField from_montgomery(const device QuarticExtensionField& value)
  {
    const QuarticExtensionField x = value;
    return from_montgomery(x);
  }

  static constexpr QuarticExtensionField from_montgomery(const thread QuarticExtensionField& value)
  {
    return QuarticExtensionField{
      BaseField::from_montgomery(value.real), BaseField::from_montgomery(value.im1),
      BaseField::from_montgomery(value.im2), BaseField::from_montgomery(value.im3)};
  }

  static constexpr ExtensionWide
  mul_wide(const thread QuarticExtensionField& xs, const thread QuarticExtensionField& ys)
  {
    if (CONFIG::nonresidue_is_negative)
      return ExtensionWide{
        BaseField::mul_wide(xs.real, ys.real) -
          (BaseField::mul_wide(xs.im1, ys.im3) + BaseField::mul_wide(xs.im2, ys.im2) +
           BaseField::mul_wide(xs.im3, ys.im1))
            .template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im1) + BaseField::mul_wide(xs.im1, ys.real) -
          (BaseField::mul_wide(xs.im2, ys.im3) + BaseField::mul_wide(xs.im3, ys.im2))
            .template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im2) + BaseField::mul_wide(xs.im1, ys.im1) +
          BaseField::mul_wide(xs.im2, ys.real) -
          BaseField::mul_wide(xs.im3, ys.im3).template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im3) + BaseField::mul_wide(xs.im1, ys.im2) +
          BaseField::mul_wide(xs.im2, ys.im1) + BaseField::mul_wide(xs.im3, ys.real)};
    else
      return ExtensionWide{
        BaseField::mul_wide(xs.real, ys.real) +
          (BaseField::mul_wide(xs.im1, ys.im3) + BaseField::mul_wide(xs.im2, ys.im2) +
           BaseField::mul_wide(xs.im3, ys.im1))
            .template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im1) + BaseField::mul_wide(xs.im1, ys.real) +
          (BaseField::mul_wide(xs.im2, ys.im3) + BaseField::mul_wide(xs.im3, ys.im2))
            .template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im2) + BaseField::mul_wide(xs.im1, ys.im1) +
          BaseField::mul_wide(xs.im2, ys.real) +
          BaseField::mul_wide(xs.im3, ys.im3).template mul_unsigned<CONFIG::nonresidue>(),
        BaseField::mul_wide(xs.real, ys.im3) + BaseField::mul_wide(xs.im1, ys.im2) +
          BaseField::mul_wide(xs.im2, ys.im1) + BaseField::mul_wide(xs.im3, ys.real)};
  }
};
