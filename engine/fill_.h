#pragma once
#include <cstdint>

#include <cuda_runtime.h>

void fill_float(
  cudaStream_t stream,
  float* out,
  uint64_t num,
  float val);

void fill_double(
  cudaStream_t stream,
  double* out,
  uint64_t num,
  double val);

