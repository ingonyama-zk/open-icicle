
#include <metal_stdlib>
using namespace metal;

template <typename F>
kernel void ntt_init_domain(
  device F* domain [[buffer(0)]],
  device const F& root_of_unity [[buffer(1)]],
  constant uint32_t& size [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= size) { return; }
  domain[gid] = root_of_unity.pow(gid);
}

template <typename F>
kernel void ntt_stage_radix2(
  device const F* input,            // Input buffer (size: n)
  device F* output,                 // Output buffer (size: n)
  device const F* twiddles,         // Twiddle factors buffer (precomputed for size >= n)
  constant uint32_t& n,             // Array size (power of 2)
  constant uint32_t& stage,         // Current stage (1-based)
  constant uint32_t& twiddle_count, // Total twiddles precomputed
  constant uint32_t& num_threads,   // Total threads in grid
  constant uint32_t& dit,
  constant uint32_t& inverse,
  constant uint32_t& batch_size,
  constant uint32_t& column_batch,
  uint gid [[thread_position_in_grid]]) // Thread ID
{
  if (gid >= num_threads) { return; }

  const uint32_t threads_per_ntt = n / 2;

  uint32_t m = 1 << stage;                     // Size of each group (2^stage)
  uint32_t half_m = m >> 1;                    // Half the group size
  uint32_t twiddle_stride = twiddle_count / m; // Stride for twiddle factors

  // Each thread processes multiple elements
  uint32_t batch_idx = column_batch ? gid % batch_size : gid / threads_per_ntt;
  uint32_t batch_offset = column_batch ? gid / batch_size : gid % threads_per_ntt;

  // compute indices ignoring the batch
  uint32_t group = batch_offset / half_m;  // Determine which group this index belongs to
  uint32_t offset = batch_offset % half_m; // Offset within the group
  uint32_t base_idx = group * m + offset;  // Compute base index
  uint32_t pair_idx = base_idx + half_m;   // Compute paired index
  // adjust for batch
  base_idx = column_batch ? base_idx * batch_size + batch_idx : base_idx + batch_idx * n;
  pair_idx = column_batch ? pair_idx * batch_size + batch_idx : pair_idx + batch_idx * n;

  // Butterfly computation
  // organized as [1 w w^2 ... w^-2 w^-1 1]
  uint32_t twiddle_idx = inverse ? twiddle_count - offset * twiddle_stride : offset * twiddle_stride;
  F twiddle = twiddles[twiddle_idx]; // Access correct twiddle factor
  F u = input[base_idx];
  F v = input[pair_idx];

  if (dit) {
    // Decimation-in-Time (DIT)
    v = twiddle * v;
    output[base_idx] = u + v;
    output[pair_idx] = u - v;
  } else {
    // Decimation-in-Frequency (DIF)
    F tmp_u = u + v;
    F tmp_v = u - v;
    output[base_idx] = tmp_u;
    output[pair_idx] = twiddle * tmp_v;
  }
}

uint32_t bit_reverse(thread uint32_t x, uint32_t num_bits)
{
  uint32_t reversed = 0;
  for (uint32_t i = 0; i < num_bits; ++i) {
    reversed = (reversed << 1) | (x & 1);
    x >>= 1;
  }
  return reversed;
}

template <typename F>
kernel void reverse_order(
  device const F* arr,
  device F* arr_reversed,
  constant uint32_t& n,            // Size of each batch
  constant uint32_t& logn,         // Log2(n)
  constant uint32_t& batch_size,   // Number of batches
  constant uint32_t& column_batch, // 1 for column batching, 0 for row batching
  uint gid [[thread_position_in_grid]])
{
  // Ensure gid is within bounds
  if (gid >= n * batch_size) { return; }

  // Compute batch and index within the batch
  uint32_t idx, batch_idx;

  if (column_batch) {
    batch_idx = gid % batch_size; // Column index
    idx = gid / batch_size;       // Row index within the column
  } else {
    batch_idx = gid / n; // Row batch index
    idx = gid % n;       // Offset within the batch
  }

  // Reverse the index within the batch
  uint32_t idx_reversed = bit_reverse(idx, logn);

  // Compute global memory offsets
  uint32_t base_idx = gid;
  uint32_t reversed_idx = column_batch ? (idx_reversed * batch_size + batch_idx) : (batch_idx * n + idx_reversed);

  // Handle in-place or out-of-place reordering
  if (arr == arr_reversed) { // In-place reordering
    if (idx < idx_reversed) {
      F val = arr[base_idx];
      arr_reversed[base_idx] = arr[reversed_idx];
      arr_reversed[reversed_idx] = val;
    }
  } else { // Out-of-place reordering
    arr_reversed[reversed_idx] = arr[base_idx];
  }
}

template <typename F>
kernel void
normalize(device F* arr, device const F& normalize_factor, constant uint32_t& n, uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  arr[gid] = arr[gid] * normalize_factor;
}

template <typename F>
kernel void batch_mul(
  device const F* in_vec,
  device const F* scalar_vec,
  device F* out_vec,
  constant uint32_t& size,         // Per batch element
  constant uint32_t& batch_size,   // Number of batches
  constant uint32_t& column_batch, // 1 for column batching, 0 for row batching
  constant uint32_t& step,         // Step for scalar index calculation
  constant uint32_t& n_scalars,    // Total number of scalars
  constant uint32_t& logn,         // Log2(size)
  constant uint32_t& bitrev,       // 1 if bit-reversed, 0 otherwise
  constant uint32_t& inverse,      // 1 for inverse, 0 for forward
  uint gid [[thread_position_in_grid]])
{
  if (gid >= size * batch_size) return;

  uint32_t idx, batch_idx;

  if (column_batch) {
    batch_idx = gid % batch_size; // Column index
    idx = gid / batch_size;       // Row index within the column
  } else {
    batch_idx = gid / size; // Row batch index
    idx = gid % size;       // Offset within the batch
  }

  // Apply bit reversal if needed
  uint32_t scalar_id = bitrev ? bit_reverse(idx, logn) : idx;

  // Compute scalar index with modular arithmetic
  uint32_t scalar_idx = inverse ? n_scalars - ((scalar_id * step) % n_scalars) : (scalar_id * step) % n_scalars;

  // Compute global memory index for input/output
  uint32_t global_idx = column_batch ? (idx * batch_size + batch_idx) : (batch_idx * size + idx);

  // Perform the multiplication
  out_vec[global_idx] = scalar_vec[scalar_idx] * in_vec[global_idx];
}
