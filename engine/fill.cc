#include "fill.h"
#include "fill_.h"

void execute_fill(
  cudaStream_t stream,
  scalar_t val,
  uint64_t n,
  void* out)
{
  if(val.dtype == dtype_t::f32) {
    fill_float(stream, (float*)out, n, val.as_f32());
  } else if(val.dtype == dtype_t::f64) {
    fill_double(stream, (double*)out, n, val.as_f64());
  } else {
    throw std::runtime_error("missing dtype case");
  }
}
