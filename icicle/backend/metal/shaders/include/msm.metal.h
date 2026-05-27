
#pragma once

#include "curve/projective.metal.h"
#include "metal_msm.h"
#include <metal_stdlib>
using namespace metal;

// this kernel initializes the buckets with zero points
// each thread initializes a different bucket
template <typename P>
kernel void initialize_buckets_kernel(device P* buckets, constant unsigned& N, uint tid [[thread_position_in_grid]])
{
  if (tid < N) buckets[tid] = P::zero(); // zero point
}

template <typename P>
kernel void single_stage_multi_reduction_kernel(
  device const P* v,
  device P* v_r,
  constant SingleStageMultiReductionParams& params,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= params.nof_threads) return;

  // we need shifted tid because we don't want to be reducing into zero buckets, this allows to skip them.
  // for write_phase==1, the read pattern is different so we don't skip over anything.
  const int shifted_tid = params.write_phase ? tid : tid + (tid + params.step) / params.step;
  const int jump = params.block_size / 2;
  const int block_id = shifted_tid / jump;
  // here the reason for shifting is the same as for shifted_tid but we skip over entire blocks which happens
  // only for write_phase=1 because of its read pattern.
  const int shifted_block_id = params.write_phase ? block_id + (block_id + params.step) / params.step : block_id;
  const int block_tid = shifted_tid % jump;
  const unsigned read_ind = params.orig_block_size * shifted_block_id + block_tid;
  const unsigned write_ind = jump * shifted_block_id + block_tid;
  const unsigned v_r_key = params.write_stride
                             ? ((write_ind / params.buckets_per_bm) * 2 + params.write_phase) * params.write_stride +
                                 write_ind % params.buckets_per_bm
                             : read_ind;
  v_r[v_r_key] = v[read_ind] + v[read_ind + jump];
}

// this kernel adds up the points in each bucket
template <typename P, typename A>
kernel void accumulate_buckets_kernel(
  device P* buckets,
  device unsigned* bucket_offsets,
  device unsigned* bucket_sizes,
  device unsigned* single_bucket_indices,
  device const unsigned* point_indices,
  device A* points,
  constant unsigned& nof_buckets,
  constant unsigned& nof_buckets_to_compute,
  constant unsigned& msm_idx_shift,
  constant unsigned& c,
  constant bool& init_buckets,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= nof_buckets_to_compute) return;
  unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
  const unsigned single_bucket_index = (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
  unsigned bucket_index = msm_index * nof_buckets + single_bucket_index;
  const unsigned bucket_offset = bucket_offsets[tid];
  const unsigned bucket_size = bucket_sizes[tid];

  P bucket;
  if (!init_buckets) bucket = buckets[bucket_index];
  for (unsigned i = 0; i < bucket_size;
       i++) { // add the relevant points starting from the relevant offset up to the bucket size
    unsigned point_ind = point_indices[bucket_offset + i];
    A point = points[point_ind];
    bucket = i || !init_buckets ? (point.is_zero() ? bucket : bucket + point)
                                : (point.is_zero() ? P::zero() : P::from_affine(point));
  }
  buckets[bucket_index] = bucket;
}

// this kernel splits the scalars into digits of size c
// each thread splits a single scalar into nof_bms digits
template <typename S>
kernel void split_scalars_kernel(
  device unsigned* buckets_indices,
  device unsigned* point_indices,
  device const S* scalars,
  constant SplitScalarsParams& params,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= params.nof_scalars) return;

  unsigned bucket_index;
  unsigned current_index;
  unsigned msm_index = tid / params.msm_size;
  S scalar = scalars[tid];
  for (unsigned bm = 0; bm < params.nof_bms_per_msm; bm++) {
    const unsigned precomputed_index = bm / params.nof_precomputed_bms_per_msm;
    const unsigned target_bm = bm % params.nof_precomputed_bms_per_msm;

    bucket_index = scalar.get_scalar_digit(bm, params.c);
    current_index = bm * params.nof_scalars + tid;

    if (bucket_index != 0) {
      buckets_indices[current_index] =
        (msm_index << (params.c + params.bm_bitsize)) | (target_bm << params.c) |
        bucket_index; // the bucket module number and the msm number are appended at the msbs
    } else {
      buckets_indices[current_index] = 0; // will be skipped
    }
    point_indices[current_index] =
      (tid * params.precompute_factor + precomputed_index) % params.points_size; // the point index is saved for later
  }
}

template <typename P>
kernel void last_pass_kernel(
  device const P* final_buckets,
  device P* final_sums,
  constant unsigned& nof_sums_per_batch,
  constant unsigned& batch_size,
  constant unsigned& nof_bms_per_batch,
  constant unsigned& orig_c,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= nof_sums_per_batch * batch_size) return;
  unsigned batch_index = tid / nof_sums_per_batch;
  unsigned batch_tid = tid % nof_sums_per_batch;
  unsigned bm_index = batch_tid / orig_c;
  unsigned bm_tid = batch_tid % orig_c;
  for (unsigned c = orig_c; c > 1;) {
    c = (c + 1) >> 1;
    bm_index <<= 1;
    if (bm_tid >= c) {
      bm_index++;
      bm_tid -= c;
    }
  }
  final_sums[tid] = final_buckets[2 * (batch_index * nof_bms_per_batch + bm_index) + 1];
}

// this kernel computes the final result using the double and add algorithm
// it is done by a single thread
template <typename P>
kernel void final_accumulation_kernel(
  device const P* final_sums,
  device P* final_results,
  constant unsigned& nof_msms,
  constant unsigned& nof_results,
  constant unsigned& c,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= nof_msms) return;
  P final_result = P::zero();
  // Note: in some cases accumulation of bm is implemented such that some bms are known to be empty. Therefore
  // skipping them.
  for (unsigned i = nof_results; i > 1; i--) {
    final_result = final_result + final_sums[i - 1 + tid * nof_results]; // add
    for (unsigned j = 0; j < c; j++)                                     // double
    {
      final_result = final_result + final_result;
    }
  }
  final_results[tid] = final_result + final_sums[tid * nof_results];
}

template <typename A, typename P>
kernel void precompute_points_kernel(
  device const A* points,
  constant unsigned& shift,
  constant unsigned& prec_factor,
  constant unsigned& count,
  device A* points_out,
  constant bool& is_montgomery,
  uint tid [[thread_position_in_grid]])
{
  if (tid >= count) return;
  A point_a = is_montgomery ? A::from_montgomery(points[tid]) : points[tid];
  P point = P::from_affine(point_a);
  points_out[tid * prec_factor] = point.to_affine();
  for (unsigned i = 1; i < prec_factor; i++) {
    for (unsigned j = 0; j < shift; j++)
      point = point.dbl();
    points_out[tid * prec_factor + i] = point.to_affine();
  }
}

// exclusive scan:
// according to
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

template <typename A>
kernel void device_exclusive_scan_upsweep(
  device uint* inout,
  constant uint& size,
  constant uint& stride,
  constant bool& last,
  constant bool& inclusive,
  uint gid [[thread_position_in_grid]])
{
  uint t = gid * 2 * stride;
  inout[size - 1 - t] += inout[size - 1 - (t + stride)];
  if (last && (t == 0)) {
    if (inclusive) inout[size] = inout[size - 1];
    inout[size - 1] = 0;
  }
}

template <typename A>
kernel void device_exclusive_scan_downsweep(
  device uint* inout, constant uint& size, constant uint& stride, uint gid [[thread_position_in_grid]])
{
  uint t = gid * 2 * stride;
  uint temp = inout[size - 1 - (t + stride)];
  inout[size - 1 - (t + stride)] = inout[size - 1 - t];
  inout[size - 1 - t] += temp;
}

// run length encode:
// according to https://erkaman.github.io/posts/cuda_rle.html

// Step 1: Mark change points in backward mask
template <typename A>
kernel void backward_mask_kernel(
  device const uint* input,
  device uint* backwardMask,
  uint id [[thread_position_in_grid]],
  uint size [[threads_per_grid]])
{
  if (id >= size) return;

  if (id == 0 || input[id] != input[id - 1]) {
    backwardMask[id] = 1;
  } else {
    backwardMask[id] = 0;
  }
}

// Step 3: Compact indices based on scanned values
template <typename A>
kernel void compact_kernel(
  device const uint* scannedMask,
  device uint* compactedMask,
  device uint* totalRuns,
  constant uint& size,
  uint id [[thread_position_in_grid]])
{
  if (id >= size) return;

  if (id == size - 1) {
    compactedMask[scannedMask[id]] = id + 1;
    *totalRuns = scannedMask[id];
  }

  if (id == 0) {
    compactedMask[0] = 0;
  } else if (scannedMask[id] != scannedMask[id - 1]) {
    compactedMask[scannedMask[id] - 1] = id;
  }
}

// Step 4: Write final RLE output
template <typename A>
kernel void write_rle_output(
  device const uint* input,
  device const uint* compactedMask,
  device uint* rle_values,
  device uint* rle_counts,
  device uint* total_runs,
  uint id [[thread_position_in_grid]])
{
  uint n = *total_runs;
  if (id >= n) return;

  rle_values[id] = input[compactedMask[id]];
  rle_counts[id] = compactedMask[id + 1] - compactedMask[id];
}