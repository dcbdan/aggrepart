#include "fill_.h"

  __global__
void _fill_float(uint64_t n, float val, float* out)
{
  uint64_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    out[i] = val;
  }
}

  __global__
void _fill_double(uint64_t n, double val, double* out)
{
  uint64_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    out[i] = val;
  }
}

void fill_float(
  cudaStream_t stream,
  float* out, uint64_t n, float val)
{
  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

  _fill_float<<<gridSize, blockSize,0,stream>>>(n, val, out);
}

void fill_double(
  cudaStream_t stream,
  double* out, uint64_t n, double val)
{
  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

  _fill_double<<<gridSize, blockSize,0,stream>>>(n, val, out);
}
