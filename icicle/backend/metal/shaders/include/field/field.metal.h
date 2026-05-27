#pragma once
#include <metal_integer>
#include "karatsuba.metal.h"
#include "arithmetic.metal.h"

template <unsigned NLIMBS>
struct storage {
  uint32_t limbs[NLIMBS];
};

/* ========================================= */

template <typename CONFIG>
class Field
{
public:
  static constexpr constant unsigned TLC = CONFIG::NLIMBS;
  static constexpr constant unsigned NBITS = CONFIG::modulus_bit_count;
  typedef storage<TLC> ff_storage;
  typedef storage<2 * TLC> ff_wide_storage;
  static constexpr constant unsigned slack_bits = 32 * TLC - NBITS;
  ff_storage storage;

  static constexpr Field zero()
  {
    Field res = {};
    return res;
  }

  static constexpr Field one()
  {
    Field res;
    res.storage.limbs[0] = 1;
    for (unsigned i = 1; i < TLC; i++) {
      res.storage.limbs[i] = 0;
    }
    return res;
  }

  bool is_zero() const
  {
    for (unsigned i = 0; i < TLC; i++) {
      if (storage.limbs[i]) return false;
    }
    return true;
  }

  Field operator+(const device Field& other) const device
  {
    const Field a = *this;
    const Field b = other;
    return a + b;
  }

  Field operator+(const thread Field& other) const device
  {
    const Field x = *this;
    return x + other;
  }

  Field operator+(const thread Field& other) const thread
  {
    Field res;
    add_limbs<TLC>(this->storage, other.storage, res.storage);

    Field res_reduced;
    const ff_storage modulus = CONFIG::modulus;
    return sub_limbs<TLC>(res.storage, modulus, res_reduced.storage) ? res : res_reduced;
  }

  Field operator/(const device Field& other) const device
  {
    const Field a = *this;
    const Field b = other;
    return a / b;
  }

  Field operator/(const thread Field& denominator) const thread
  {
    Field inv = inverse(denominator);
    return *this * inv;
  }

  bool operator==(const device Field& other) const device
  {
    // move from device to thread memory
    const Field a = *this;
    const Field b = other;
    return a == b;
  }

  bool operator==(const thread Field& other) const thread
  {
    for (unsigned i = 0; i < TLC; i++) {
      if (storage.limbs[i] != other.storage.limbs[i]) return false;
    }
    return true;
  }

  Field operator-(const device Field& other) const device
  {
    const Field a = *this;
    const Field b = other;
    return a - b;
  }

  Field operator-(const thread Field& y) const device
  {
    const Field x = *this;
    return x - y;
  }

  Field operator-(const thread Field& other) const thread
  {
    Field res;
    uint32_t carry = sub_limbs<TLC>(this->storage, other.storage, res.storage);
    if (carry == 0) return res;

    const ff_storage modulus = CONFIG::modulus;
    add_limbs<TLC>(res.storage, modulus, res.storage);

    return res;
  }

  Field operator*(const device Field& other) const device
  {
    const Field a = *this;
    const Field b = other;
    return a * b;
  }

  Field operator*(const thread Field& scalar) const device
  {
    const Field element = *this;
    return scalar * element;
  }

  Field operator*(const thread Field& other) const thread
  {
    Wide xy = mul_wide(*this, other); // full mult
    return xy.reduce();               // reduce mod p
  }

  bool operator<(const thread Field& ys) const
  {
    ff_storage dummy = {};
    uint32_t carry = sub_limbs<TLC>(this->storage, ys.storage, dummy);
    return carry;
  }

  static constexpr Field from_uint32(uint32_t x)
  {
    Field res;
    res.storage.limbs[0] = x;
    for (unsigned i = 1; i < TLC; i++) {
      res.storage.limbs[i] = 0;
    }
    return res;
  }

  Field pow(uint32_t exp) const device
  {
    Field base = *this;
    Field res = Field::one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }

  constexpr Field sqr() const
  {
    // TODO: change to a more efficient squaring
    return *this * *this;
  }

  constexpr Field neg() const
  {
    const ff_storage modulus = CONFIG::modulus;
    Field rs = {};
    sub_limbs<TLC>(modulus, this->storage, rs.storage);
    return rs;
  }

  // Assumes the number is even!

  static inline Field div2(const thread Field& xs)
  {
    const thread uint32_t* x = xs.storage.limbs;
    Field rs = zero(); // Initialize the result field
    thread uint32_t* r = rs.storage.limbs;

    for (uint i = 0; i < TLC - 1; i++) {
      r[i] = (x[i] >> 1) | (x[i + 1] << 31);
    }
    r[TLC - 1] = x[TLC - 1] >> 1;

    // TODO: all sub_modulus to ensure the result is reduced
    // return sub_modulus(rs);
    return rs;
  }

  static inline bool is_even(const thread Field& xs) { return ~(xs.storage.limbs[0]) & 1; }

  static inline bool is_odd(const thread Field& xs) { return (xs.storage.limbs[0]) & 1; }

  static Field inverse(const device Field& xs)
  {
    Field x = xs;
    return inverse(x);
  }

  static Field inverse(const thread Field& xs)
  {
    Field u = xs;
    if (u.is_zero()) return zero();
    const ff_storage modulus = CONFIG::modulus;
    const Field one = Field::one();

    Field v = Field{CONFIG::modulus};
    Field b = one;
    Field c = zero();

    while (!(u == one) && !(v == one)) {
      while (is_even(u)) {
        u = div2(u);
        if (is_odd(b)) add_limbs<TLC>(b.storage, modulus, b.storage);
        b = div2(b);
      }
      while (is_even(v)) {
        v = div2(v);
        if (is_odd(c)) add_limbs<TLC>(c.storage, modulus, c.storage);
        c = div2(c);
      }
      if (v < u) {
        u = u - v;
        b = b - c;
      } else {
        v = v - u;
        c = c - b;
      }
    }
    return (u == one) ? b : c;
  }

  static constexpr Field to_montgomery(const device Field& value)
  {
    const Field x = value;
    return to_montgomery(x);
  }

  static constexpr Field to_montgomery(const thread Field& value)
  {
    const Field m_r = Field{CONFIG::montgomery_r};
    return value * m_r;
  }

  static constexpr Field from_montgomery(const device Field& value)
  {
    const Field x = value;
    return from_montgomery(x);
  }

  static constexpr Field from_montgomery(const thread Field& value)
  {
    const Field m_r_inv = Field{CONFIG::montgomery_r_inv};
    return value * m_r_inv;
  }

  unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    const uint32_t limb_lsb_idx = (digit_num * digit_width) / 32;
    const uint32_t shift_bits = (digit_num * digit_width) % 32;
    unsigned rv = storage.limbs[limb_lsb_idx] >> shift_bits;
    if ((shift_bits + digit_width > 32) && (limb_lsb_idx + 1 < TLC)) {
      rv += storage.limbs[limb_lsb_idx + 1] << (32 - shift_bits);
    }
    rv &= ((1 << digit_width) - 1);
    return rv;
  }

  template <thread const Field& M>
  constexpr Field mul_const() const
  {
    Field mul = M;
    bool is_u32 = true;
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
      is_u32 &= (mul.storage.limbs[i] == 0);

    if (is_u32) return this->mul_unsigned<M.storage.limbs[0]>();
    return mul * *this;
  }

  template <uint32_t M>
  constexpr Field mul_unsigned() const
  {
    Field rs = {};
    Field temp = *this;
    bool is_zero = true;
#pragma unroll
    for (unsigned i = 0; i < 32; i++) {
      if (M & (1 << i)) {
        rs = is_zero ? temp : (rs + temp);
        is_zero = false;
      }
      if (M & ((1 << (31 - i - 1)) << (i + 1))) break;
      temp = temp + temp;
    }
    return rs;
  }

  static constexpr Field from(const device uint32_t& value)
  {
    Field scalar{};
    scalar.storage.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      scalar.storage.limbs[i] = 0;
    }
    return scalar;
  }

  struct Wide {
    ff_wide_storage storage;

    static constexpr Wide from_field(const device Field& xs)
    {
      Wide out{};
      for (unsigned i = 0; i < TLC; i++)
        out.storage.limbs[i] = xs.storage.limbs[i];
      return out;
    }

    constexpr Field get_lower() const
    {
      Field out{};
#pragma unroll
      for (unsigned i = 0; i < TLC; i++)
        out.storage.limbs[i] = this->storage.limbs[i];
      return out;
    }

    constexpr Field get_higher() const
    {
      Field out{};
#pragma unroll
      for (unsigned i = 0; i < TLC; i++)
        out.storage.limbs[i] = this->storage.limbs[i + TLC];
      return out;
    }

    constexpr Field get_higher_with_slack() const
    {
      Field out{};
      for (unsigned i = 0; i < TLC; i++) {
        out.storage.limbs[i] = (this->storage.limbs[i + TLC] << 2 * slack_bits) +
                               (this->storage.limbs[i + TLC - 1] >> (32 - 2 * slack_bits));
      }
      return out;
    }

    constexpr Field reduce() const
    {
      // `xs` is left-shifted by `2 * slack_bits` and higher half is written to `xs_hi`
      Field xs_hi = this->get_higher_with_slack();
      Wide l = {};
      const ff_storage m = CONFIG::m;
      karatsuba::multiply_msb_raw<ff_storage, ff_wide_storage, TLC>(xs_hi.storage, m, l.storage); // MSB mult by `m`
      Field l_hi = l.get_higher();
      Field r = {};
      Field xs_lo = this->get_lower();
      // Here we need to compute the lsb of `xs - l \cdot p` and to make use of fused multiply-and-add, we rewrite it as
      // `xs + l \cdot (2^{32 \cdot TLC}-p)` which is the same as original (up to higher limbs which we don't care
      // about).
      const ff_storage neg_modulus = CONFIG::neg_modulus;
      karatsuba::multiply_and_add_lsb_neg_modulus_raw<ff_storage, ff_wide_storage, TLC>(
        l_hi.storage, neg_modulus, xs_lo.storage, r.storage);
      ff_storage r_reduced = {};
      uint32_t carry = 0;
      // As mentioned, either 2 or 1 reduction can be performed depending on the field in question.
      if (CONFIG::num_of_reductions == 2) {
        const ff_storage modulus_2 = CONFIG::modulus_2;
        carry = sub_limbs<TLC>(r.storage, modulus_2, r_reduced);
        if (carry == 0) r = Field{r_reduced};
      }
      const ff_storage modulus = CONFIG::modulus;
      carry = sub_limbs<TLC>(r.storage, modulus, r_reduced);
      if (carry == 0) r = Field{r_reduced};

      return r;
    }

    constexpr Wide sub_modulus_squared() const
    {
      const ff_wide_storage modulus = CONFIG::modulus_squared;
      Wide rs = {};
      return sub_limbs<2 * TLC>(this->storage, modulus, rs.storage) ? *this : rs;
    }

    constexpr Wide neg() const
    {
      const ff_wide_storage modulus = CONFIG::modulus_squared;
      Wide rs = {};
      sub_limbs<2 * TLC>(modulus, this->storage, rs.storage);
      return rs;
    }

    Wide operator+(const thread Wide& y) const device
    {
      const Wide x = *this;
      return x + y;
    }

    Wide operator+(const device Wide& other) const device
    {
      const Wide a = *this;
      const Wide b = other;
      return a + b;
    }

    Wide operator+(thread const Wide& ys) const
    {
      Wide rs = {};
      add_limbs<2 * TLC>(this->storage, ys.storage, rs.storage);
      return rs.sub_modulus_squared();
    }

    Wide operator-(const device Wide& other) const device
    {
      const Wide a = *this;
      const Wide b = other;
      return a - b;
    }

    Wide operator-(const thread Wide& y) const device
    {
      const Wide x = *this;
      return x - y;
    }

    Wide operator-(thread const Wide& ys) const
    {
      Wide rs = {};
      uint32_t carry = sub_limbs<2 * TLC>(this->storage, ys.storage, rs.storage);
      if (carry == 0) return rs;
      const ff_wide_storage modulus = CONFIG::modulus_squared;
      add_limbs<2 * TLC>(rs.storage, modulus, rs.storage);
      return rs;
    }

    template <uint32_t M>
    constexpr Wide mul_unsigned() const
    {
      Wide rs = {};
      Wide temp = *this;
      bool is_zero = true;
#pragma unroll
      for (unsigned i = 0; i < 32; i++) {
        if (M & (1 << i)) {
          rs = is_zero ? temp : (rs + temp);
          is_zero = false;
        }
        if (M & ((1 << (31 - i - 1)) << (i + 1))) break;
        temp = temp + temp;
      }
      return rs;
    }
  };

  template <unsigned NLIMBS>
  static constexpr uint32_t
  add_limbs(const thread ::storage<NLIMBS>& x, const thread ::storage<NLIMBS>& y, thread ::storage<NLIMBS>& r)
  {
    const thread uint32_t* xs = x.limbs;
    const thread uint32_t* ys = y.limbs;
    thread uint32_t* rs = r.limbs;

    return arithmetic::add_u32<NLIMBS>(xs, ys, rs);
  }

  template <unsigned NLIMBS>
  static constexpr uint32_t
  sub_limbs(const thread ::storage<NLIMBS>& x, const thread ::storage<NLIMBS>& y, thread ::storage<NLIMBS>& r)
  {
    const thread uint32_t* xs = x.limbs;
    const thread uint32_t* ys = y.limbs;
    thread uint32_t* rs = r.limbs;

    return arithmetic::sub_u32<NLIMBS>(xs, ys, rs);
  }

  static constexpr Wide mul_wide(const thread Field& xs, const thread Field& ys)
  {
    Wide rs = {};
    ff_storage x = xs.storage;
    ff_storage y = ys.storage;
    karatsuba::multiply_raw<ff_storage, ff_wide_storage, TLC>(x, y, rs.storage);
    return rs;
  }

  constexpr Wide sqr_wide() const
  {
    // TODO: change to a more efficient squaring
    return mul_wide(*this, *this);
  }

  // Compute multiplication by performing single round of Montgomery reduction
  template <const uint32_t MU>
  Field mont_mul(const thread Field& other) const thread
  {
    // constexpr static u256 mont_mul(const u256 a, const u256 b) {
    constexpr uint64_t NUM_LIMBS = TLC;
    ::storage<NUM_LIMBS> t = {};
    ::storage<2> t_extra = {};

    const ff_storage q = CONFIG::modulus;

    uint64_t i = NUM_LIMBS;

    while (i > 0) {
      i -= 1;
      uint64_t c = 0;

      uint64_t cs = 0;
      uint64_t j = NUM_LIMBS;
      while (j > 0) {
        j -= 1;
        cs = (uint64_t)t.limbs[NUM_LIMBS - 1 - j] +
             (uint64_t)this->storage.limbs[NUM_LIMBS - 1 - j] * (uint64_t)other.storage.limbs[NUM_LIMBS - 1 - i] + c;
        c = cs >> 32;
        t.limbs[NUM_LIMBS - 1 - j] = (uint32_t)((cs << 32) >> 32);
      }

      cs = (uint64_t)t_extra.limbs[0] + c;
      t_extra.limbs[1] = (uint32_t)(cs >> 32);
      t_extra.limbs[0] = (uint32_t)((cs << 32) >> 32);

      uint64_t m = (((uint64_t)t.limbs[0] * MU) << 32) >> 32;

      c = ((uint64_t)t.limbs[0] + m * (uint64_t)q.limbs[0]) >> 32;

      j = NUM_LIMBS - 1;
      while (j > 0) {
        j -= 1;
        cs = (uint64_t)t.limbs[NUM_LIMBS - 1 - j] + m * (uint64_t)q.limbs[NUM_LIMBS - j - 1] + c;
        c = cs >> 32;
        t.limbs[NUM_LIMBS - 1 - (j + 1)] = (uint32_t)((cs << 32) >> 32);
      }

      cs = (uint64_t)t_extra.limbs[0] + c;
      c = cs >> 32;
      t.limbs[NUM_LIMBS - 1] = (uint32_t)((cs << 32) >> 32);

      t_extra.limbs[0] = t_extra.limbs[1] + (uint32_t)c;
    }

    Field res{t};

    uint64_t overflow = t_extra.limbs[1] > 0;

    Field res_reduced;
    auto carry = sub_limbs<TLC>(res.storage, q, res_reduced.storage);

    if (overflow || (!carry)) { return res_reduced; }

    return res;
  }
};