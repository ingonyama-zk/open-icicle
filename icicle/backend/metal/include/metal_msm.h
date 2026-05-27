struct SingleStageMultiReductionParams {
  unsigned orig_block_size;
  unsigned block_size;
  unsigned write_stride;
  unsigned buckets_per_bm;
  unsigned write_phase;
  unsigned step;
  unsigned nof_threads;
};

struct SplitScalarsParams {
  unsigned nof_scalars;
  unsigned points_size;
  unsigned msm_size;
  unsigned nof_bms_per_msm;
  unsigned bm_bitsize;
  unsigned c;
  unsigned precompute_factor;
  unsigned nof_precomputed_bms_per_msm;
};