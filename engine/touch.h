#pragma once
#include "../repr/touch.h"

#include <cuda_runtime.h>

void execute_touch(
  cudaStream_t stream,
  touch_t const& touch,
  void* out,
  void const* inn);
