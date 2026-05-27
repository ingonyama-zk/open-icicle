
#pragma once

#include "affine.metal.h"

template <typename BaseField, class ScalarField, typename Gen>
class Projective
{
  friend Affine<BaseField>;

public:
  static constant constexpr unsigned SCALAR_FF_NBITS = ScalarField::NBITS;
  static constant constexpr unsigned FF_NBITS = BaseField::NBITS;

  BaseField x;
  BaseField y;
  BaseField z;

  static Projective zero() { return {BaseField::zero(), BaseField::one(), BaseField::zero()}; }

  Affine<BaseField> to_affine() const
  {
    BaseField denom = BaseField::inverse(this->z);
    return {x * denom, y * denom};
  }

  static Projective from_affine(thread const Affine<BaseField>& point)
  {
    return point.is_zero() ? zero() : Projective{point.x, point.y, BaseField::one()};
  }

  static Projective to_montgomery(device const Projective& point)
  {
    const Projective x = point;
    return to_montgomery(x);
  }

  static Projective to_montgomery(thread const Projective& point)
  {
    return {BaseField::to_montgomery(point.x), BaseField::to_montgomery(point.y), BaseField::to_montgomery(point.z)};
  }

  static Projective from_montgomery(device const Projective& point)
  {
    const Projective x = point;
    return from_montgomery(x);
  }

  static Projective from_montgomery(thread const Projective& point)
  {
    return {
      BaseField::from_montgomery(point.x), BaseField::from_montgomery(point.y), BaseField::from_montgomery(point.z)};
  }

  static Projective generator() { return {Gen::gen_x, Gen::gen_y, BaseField::one()}; }

  Projective neg() const { return {this->x, this->y.neg(), this->z}; }

  Projective dbl() const
  {
    const BaseField X = this->x;
    const BaseField Y = this->y;
    const BaseField Z = this->z;

    // TODO: Change to efficient dbl once implemented for field.cuh
    BaseField t0 = Y.sqr();                                                      // 1. t0 ← Y · Y
    BaseField Z3 = t0 + t0;                                                      // 2. Z3 ← t0 + t0
    Z3 = Z3 + Z3;                                                                // 3. Z3 ← Z3 + Z3
    Z3 = Z3 + Z3;                                                                // 4. Z3 ← Z3 + Z3
    BaseField t1 = Y * Z;                                                        // 5. t1 ← Y · Z
    BaseField t2 = Z.sqr();                                                      // 6. t2 ← Z · Z
    t2 = (t2 * BaseField{Gen::weierstrass_b}).template mul_unsigned<3>();         // 7. t2 ← 3b · t2
    BaseField X3 = t2 * Z3;                                                      // 8. X3 ← t2 · Z3
    BaseField Y3 = t0 + t2;                                                      // 9. Y3 ← t0 + t2
    Z3 = t1 * Z3;                                                                // 10. Z3 ← t1 · Z3
    t1 = t2 + t2;                                                                // 11. t1 ← t2 + t2
    t2 = t1 + t2;                                                                // 12. t2 ← t1 + t2
    t0 = t0 - t2;                                                                // 13. t0 ← t0 − t2
    Y3 = t0 * Y3;                                                                // 14. Y3 ← t0 · Y3
    Y3 = X3 + Y3;                                                                // 15. Y3 ← X3 + Y3
    t1 = X * Y;                                                                  // 16. t1 ← X · Y
    X3 = t0 * t1;                                                                // 17. X3 ← t0 · t1
    X3 = X3 + X3;                                                                // 18. X3 ← X3 + X3
    return {X3, Y3, Z3};
  }

  Projective operator+(device const Projective& p2) const device
  {
    Projective local = *this;
    Projective other = p2;
    return local + other;
  }

  Projective operator+(thread const Projective& p2) const device
  {
    Projective local = *this;
    return local + p2;
  }

  Projective operator+(device const Projective& p2) const thread
  {
    Projective other = p2;
    return *this + other;
  }

  Projective operator+(thread const Projective& p2) const
  {
    const BaseField X1 = this->x;                            //                   < 2
    const BaseField Y1 = this->y;                            //                   < 2
    const BaseField Z1 = this->z;                            //                   < 2
    const BaseField X2 = p2.x;                               //                   < 2
    const BaseField Y2 = p2.y;                               //                   < 2
    const BaseField Z2 = p2.z;                               //                   < 2
    const BaseField t00 = X1.template mont_mul<Gen::mu>(X2); // t00 ← X1 · X2     < 2
    const BaseField t01 = Y1.template mont_mul<Gen::mu>(Y2); // t01 ← Y1 · Y2     < 2
    const BaseField t02 = Z1.template mont_mul<Gen::mu>(Z2); // t02 ← Z1 · Z2     < 2
    // const BaseField t00 = X1 * X2;   // t00 ← X1 · X2     < 2
    // const BaseField t01 = Y1 * Y2;   // t01 ← Y1 · Y2     < 2
    // const BaseField t02 = Z1 * Z2;   // t02 ← Z1 · Z2     < 2
    const BaseField t03 = X1 + Y1;                             // t03 ← X1 + Y1     < 4
    const BaseField t04 = X2 + Y2;                             // t04 ← X2 + Y2     < 4
    const BaseField t05 = t03.template mont_mul<Gen::mu>(t04); // t03 ← t03 · t04   < 3
    // const BaseField t05 = t03 * t04; // t03 ← t03 · t04   < 3
    const BaseField t06 = t00 + t01;                           // t06 ← t00 + t01   < 4
    const BaseField t07 = t05 - t06;                           // t05 ← t05 − t06   < 2
    const BaseField t08 = Y1 + Z1;                             // t08 ← Y1 + Z1     < 4
    const BaseField t09 = Y2 + Z2;                             // t09 ← Y2 + Z2     < 4
    const BaseField t10 = t08.template mont_mul<Gen::mu>(t09); // t10 ← t08 · t09   < 3
    // const BaseField t10 = t08 * t09; // t10 ← t08 · t09   < 3
    const BaseField t11 = t01 + t02;                           // t11 ← t01 + t02   < 4
    const BaseField t12 = t10 - t11;                           // t12 ← t10 − t11   < 2
    const BaseField t13 = X1 + Z1;                             // t13 ← X1 + Z1     < 4
    const BaseField t14 = X2 + Z2;                             // t14 ← X2 + Z2     < 4
    const BaseField t15 = t13.template mont_mul<Gen::mu>(t14); // t15 ← t13 · t14   < 3
    // const BaseField t15 = t13 * t14; // t15 ← t13 · t14   < 3
    const BaseField t16 = t00 + t02; // t16 ← t00 + t02   < 4
    const BaseField t17 = t15 - t16; // t17 ← t15 − t16   < 2
    const BaseField t18 = t00 + t00; // t18 ← t00 + t00   < 2
    const BaseField t19 = t18 + t00; // t19 ← t18 + t00   < 2
    const BaseField t20 =            // t20 ← b3 · t02    < 2
      (t02 * BaseField{Gen::weierstrass_b}).template mul_unsigned<3>();
    const BaseField t21 = t01 + t20; // t21 ← t01 + t20   < 2
    const BaseField t22 = t01 - t20; // t22 ← t01 − t20   < 2
    const BaseField t23 =            // t23 ← b3 · t17    < 2
      (t17 * BaseField{Gen::weierstrass_b}).template mul_unsigned<3>();
    // const auto t24 = BaseField::mul_wide(t12, t23); // t24 ← t12 · t23   < 2
    // const auto t25 = BaseField::mul_wide(t07, t22); // t25 ← t07 · t22   < 2
    // const BaseField X3 = (t25 - t24).reduce();      // X3 ← t25 − t24    < 2
    const BaseField X3 = t07.template mont_mul<Gen::mu>(t22) - t12.template mont_mul<Gen::mu>(t23);
    // const auto t27 = BaseField::mul_wide(t23, t19); // t27 ← t23 · t19   < 2
    // const auto t28 = BaseField::mul_wide(t22, t21); // t28 ← t22 · t21   < 2
    // const BaseField Y3 = (t28 + t27).reduce();      // Y3 ← t28 + t27    < 2
    const BaseField Y3 = t23.template mont_mul<Gen::mu>(t19) + t22.template mont_mul<Gen::mu>(t21);
    // const auto t30 = BaseField::mul_wide(t19, t07); // t30 ← t19 · t07   < 2
    // const auto t31 = BaseField::mul_wide(t21, t12); // t31 ← t21 · t12   < 2
    // const BaseField Z3 = (t31 + t30).reduce();      // Z3 ← t31 + t30    < 2
    const BaseField Z3 = t19.template mont_mul<Gen::mu>(t07) + t21.template mont_mul<Gen::mu>(t12);
    return {X3, Y3, Z3};
    // return {X1, Y2, Z1};
    // return {X1.template mont_mul<Gen::mu>(X2), Y2, Z1};
  }

  Projective operator-(thread const Projective& p2) const { return *this + p2.neg(); }

  Projective operator+(thread const Affine<BaseField>& p2) const
  {
    // const BaseField temp = BaseField::one() + BaseField::one();
    const BaseField X1 = this->x;                                          //                   < 2
    const BaseField Y1 = this->y;                                          //                   < 2
    const BaseField Z1 = this->z;                                          //                   < 2
    const BaseField X2 = p2.x;                                             //                   < 2
    const BaseField Y2 = p2.y;                                             //                   < 2
    const BaseField t00 = X1.template mont_mul<Gen::mu>(X2);               // t00 ← X1 · X2     < 2
    const BaseField t01 = Y1.template mont_mul<Gen::mu>(Y2);               // t01 ← Y1 · Y2     < 2
    const BaseField t02 = Z1.template mont_mul<Gen::mu>(BaseField::one()); // t02 ← Z1          < 2
    // const BaseField t00 = X1 * X2 * temp;               // t00 ← X1 · X2     < 2
    // const BaseField t01 = Y1 * Y2 * temp;               // t01 ← Y1 · Y2     < 2
    // const BaseField t02 = Z1 * temp;                    // t02 ← Z1          < 2
    const BaseField t03 = X1 + Y1;                             // t03 ← X1 + Y1     < 4
    const BaseField t04 = X2 + Y2;                             // t04 ← X2 + Y2     < 4
    const BaseField t05 = t03.template mont_mul<Gen::mu>(t04); // t03 ← t03 · t04   < 3
    // const BaseField t05 = t03 * t04 * temp; // t03 ← t03 · t04   < 3
    const BaseField t06 = t00 + t01;                           // t06 ← t00 + t01   < 4
    const BaseField t07 = t05 - t06;                           // t05 ← t05 − t06   < 2
    const BaseField t08 = Y1 + Z1;                             // t08 ← Y1 + Z1     < 4
    const BaseField t09 = Y2 + BaseField::one();               // t09 ← Y2 + 1      < 4
    const BaseField t10 = t08.template mont_mul<Gen::mu>(t09); // t10 ← t08 · t09   < 3
    // const BaseField t10 = t08 * t09 * temp; // t10 ← t08 · t09   < 3
    const BaseField t11 = t01 + t02;                           // t11 ← t01 + t02   < 4
    const BaseField t12 = t10 - t11;                           // t12 ← t10 − t11   < 2
    const BaseField t13 = X1 + Z1;                             // t13 ← X1 + Z1     < 4
    const BaseField t14 = X2 + BaseField::one();               // t14 ← X2 + 1      < 4
    const BaseField t15 = t13.template mont_mul<Gen::mu>(t14); // t15 ← t13 · t14   < 3
    // const BaseField t15 = t13 * t14 * temp; // t15 ← t13 · t14   < 3
    const BaseField t16 = t00 + t02; // t16 ← t00 + t02   < 4
    const BaseField t17 = t15 - t16; // t17 ← t15 − t16   < 2
    const BaseField t18 = t00 + t00; // t18 ← t00 + t00   < 2
    const BaseField t19 = t18 + t00; // t19 ← t18 + t00   < 2
    const BaseField t20 =            // t20 ← b3 · t02    < 2
      (t02 * BaseField{Gen::weierstrass_b}).template mul_unsigned<3>();
    const BaseField t21 = t01 + t20; // t21 ← t01 + t20   < 2
    const BaseField t22 = t01 - t20; // t22 ← t01 − t20   < 2
    const BaseField t23 =            // t23 ← b3 · t17    < 2
      (t17 * BaseField{Gen::weierstrass_b}).template mul_unsigned<3>();
    // const auto t24 = BaseField::mul_wide(t12, t23); // t24 ← t12 · t23   < 2
    // const auto t25 = BaseField::mul_wide(t07, t22); // t25 ← t07 · t22   < 2
    // const BaseField X3 = (t25 - t24).reduce();      // X3 ← t25 − t24    < 2
    const BaseField X3 = t07.template mont_mul<Gen::mu>(t22) - t12.template mont_mul<Gen::mu>(t23);
    // const BaseField X3 = t07 * t22 * temp - t12 * t23 * temp;
    // const auto t27 = BaseField::mul_wide(t23, t19); // t27 ← t23 · t19   < 2
    // const auto t28 = BaseField::mul_wide(t22, t21); // t28 ← t22 · t21   < 2
    // const BaseField Y3 = (t28 + t27).reduce();      // Y3 ← t28 + t27    < 2
    const BaseField Y3 = t23.template mont_mul<Gen::mu>(t19) + t22.template mont_mul<Gen::mu>(t21);
    // const BaseField Y3 = t23 * t19 * temp + t22 * t21 * temp;
    // const auto t30 = BaseField::mul_wide(t19, t07); // t30 ← t19 · t07   < 2
    // const auto t31 = BaseField::mul_wide(t21, t12); // t31 ← t21 · t12   < 2
    // const BaseField Z3 = (t31 + t30).reduce();      // Z3 ← t31 + t30    < 2
    const BaseField Z3 = t19.template mont_mul<Gen::mu>(t07) + t21.template mont_mul<Gen::mu>(t12);
    // const BaseField Z3 = t19 * t07 * temp + t21 * t12 * temp;
    return {X3, Y3, Z3};
    // return {X1, Y2, Z1};
  }

  Projective operator-(thread const Affine<BaseField>& p2) const { return *this + p2.neg(); }

  Projective operator*(ScalarField scalar) const
  {
    // Precompute points: P, 2P, ..., (2^window_size - 1)P
    constexpr unsigned window_size =
      4; // 4 seems fastest. Optimum is minimizing EC add and depends on the field size. for 256b it's 4.
    constexpr unsigned table_size = (1 << window_size) - 1; // 2^window_size-1
    Projective table[table_size];
    table[0] = this;
    for (int i = 1; i < table_size; ++i) {
      table[i] = table[i - 1] + this; // Compute (i+1)P
    }

    Projective res = zero();

    const int nof_windows = (ScalarField::NBITS + window_size - 1) / window_size;
    bool res_is_not_zero = false;
    for (int w = nof_windows - 1; w >= 0; w -= 1) {
      // Extract the next window_size bits from the scalar
      unsigned window = scalar.get_scalar_digit(w, window_size);

      // Double the result window_size times
      for (int j = 0; res_is_not_zero && j < window_size; ++j) {
        res = res.dbl(); // Point doubling
      }

      // Add the precomputed value if window is not zero
      if (window != 0) {
        res = res + table[window - 1]; // Add the precomputed point
        res_is_not_zero = true;
      }
    }
    return res;
  }

  friend Projective operator*(ScalarField scalar, thread const Projective& point) { return point * scalar; }

  bool operator==(thread const Projective& p2)
  {
    return (this->x * p2.z == p2.x * this->z) && (this->y * p2.z == p2.y * this->z);
  }

  bool operator!=(thread const Projective& p2) const { return !(*this == p2); }

  bool is_zero() const
  {
    return this->x == BaseField::zero() && this->y != BaseField::zero() && this->z == BaseField::zero();
  }

  bool is_on_curve() const
  {
    if (this->is_zero()) return true;
    const BaseField ls = (this->z.sqr() * this->z) * BaseField{Gen::weierstrass_b} + this->x.sqrt() * this->x;
    const BaseField rs = this->z * this->y.sqr();
    return this->z != BaseField::zero() && ls == rs;
  }
};