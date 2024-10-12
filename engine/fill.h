#pragma once
#include "../repr/scalar.h"

#include <cuda_runtime.h>

void execute_fill(
  cudaStream_t stream,
  scalar_t val,
  uint64_t n,
  void* out);
