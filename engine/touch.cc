#include "touch.h"
#include "touch_.h"

void touch1(touch_t::dim_t const& t0, void* out,
            void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch1_dispatch(out, inn, t0.offset_inn,
  t0.offset_out, t0.size, t0.d_inn, t0.d_out,
  stream, choice, dtype_info);
}

void touch2(touch_t::dim_t const& t0, touch_t::dim_t const& t1,
            void* out, void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch2_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t0.offset_out, t1.offset_out,
  t0.size, t1.size, t1.d_inn, t1.d_out,
  stream, choice, dtype_info);
}

void touch3(touch_t::dim_t const& t0, touch_t::dim_t const& t1,
            touch_t::dim_t const& t2, void* out, void const* inn,
            cudaStream_t stream, int choice, int dtype_info) {
  touch3_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t2.offset_inn, t0.offset_out,
  t1.offset_out, t2.offset_out,t0.size,
  t1.size, t2.size, t1.d_inn, t1.d_out,
  t2.d_inn, t2.d_out, stream, choice, dtype_info);
}

void touch4(touch_t::dim_t const& t0, touch_t::dim_t const& t1,
            touch_t::dim_t const& t2, touch_t::dim_t const& t3,
            void* out, void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch4_dispatch(out, inn, t0.offset_inn, t1.offset_inn,
  t2.offset_inn, t3.offset_inn,t0.offset_out, t1.offset_out,
  t2.offset_out, t3.offset_out,t0.size, t1.size, t2.size,
  t3.size, t1.d_inn, t1.d_out, t2.d_inn, t2.d_out, t3.d_inn,
  t3.d_out, stream, choice, dtype_info);
}

#define _touch_lambda_1(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch1(ts[0], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_2(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch2(ts[0], ts[1], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_3(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_4(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch4(ts[0], ts[1], ts[2], ts[3],out, inn, stream, choice, dtype_info); \
}


#define _touch_dispatch(i) \
  [&]() -> touch_kernel_t { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_##i(1); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_##i(2); \
      } else if(c == castable_t::min) { \
        return _touch_lambda_##i(3); \
      } else if(c == castable_t::max) { \
        return  _touch_lambda_##i(4); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_##i(0); \
    } \
  }()

void execute_touch(
  cudaStream_t stream,
  touch_t const& touch,
  void* out,
  void const* inn)
{
  auto const& ts = touch.dims;
  auto const& dtype = touch.dtype;

  int dtype_info;
  if(dtype == dtype_t::f32) {
    dtype_info = 0;
  } else if(dtype == dtype_t::f64) {
    dtype_info = 1;
  }

  int choice = 0;
  if(touch.castable) {
    castable_t const& c = touch.castable.value();
    if(c == castable_t::add) {
      choice = 1;
    } else if(c == castable_t::min) {
      choice = 2;
    } else if(c == castable_t::max) {
      choice = 3;
    } else {
      throw std::runtime_error("castable should not reach");
    }
  }

  if(ts.size() == 1) {
    touch1(ts[0], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 2) {
    touch2(ts[0], ts[1], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 3) {
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 4) {
    touch4(ts[0], ts[1], ts[2], ts[3], out, inn, stream, choice, dtype_info);
  } else {
    throw std::runtime_error("touch kernel not implemented");
  }
}

