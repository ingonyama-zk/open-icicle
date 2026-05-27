#pragma once

template <class BaseField>
class Affine
{
public:
  BaseField x;
  BaseField y;

  Affine neg() const { return {x, y.neg()}; }

  static Affine zero() { return {BaseField::zero(), BaseField::zero()}; }

  static Affine to_montgomery(device const Affine& point)
  {
    const Affine x = point;
    return to_montgomery(x);
  }

  static Affine to_montgomery(thread const Affine& point)
  {
    return {BaseField::to_montgomery(point.x), BaseField::to_montgomery(point.y)};
  }

  static Affine from_montgomery(device const Affine& point)
  {
    const Affine x = point;
    return from_montgomery(x);
  }

  static Affine from_montgomery(thread const Affine& point)
  {
    return {BaseField::from_montgomery(point.x), BaseField::from_montgomery(point.y)};
  }

  bool is_zero() const { return x.is_zero() && y.is_zero(); }

  bool operator==(thread const Affine& ys) const { return (this->x == ys.x) && (this->y == ys.y); }

  bool operator!=(thread const Affine& ys) const { return !(*this == ys); }
};