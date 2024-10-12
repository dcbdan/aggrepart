#include "scalar.h"

uint64_t dtype_size(dtype_t d) {
  if(d == dtype_t::f32) {
    return sizeof(float);
  } else if(d == dtype_t::f64) {
    return sizeof(double);
  } else {
    throw std::runtime_error("should not occur");
  }
}

scalar_t scalar_t::make_zero(dtype_t dtype) {
  scalar_t ret {
    .dtype = dtype
  };
  std::fill(ret.data, ret.data + _scalar_data_size, 0);
  return ret;
}

scalar_t scalar_t::make_zero(castable_t castable, dtype_t dtype) {
  scalar_t ret = make_zero(dtype);

  if(castable == castable_t::add) {
    return make_zero(dtype);
  }
  if(castable == castable_t::max) {
    return make_negative_inf(dtype);
  }
  if(castable == castable_t::min) {
    return make_inf(dtype);
  }

  throw std::runtime_error("missing castable case");
}

double _f64_inf() {
  static_assert(std::numeric_limits<double>::is_iec559, "for inf");
  return std::numeric_limits<double>::infinity();
}
double _f64_ninf() {
  static_assert(std::numeric_limits<double>::is_iec559, "for ninf");
  double ret = - std::numeric_limits<double>::infinity();
  return ret;
}

static
float const&
f32_inf() {
  static float ret(_f64_inf());
  return ret;
}
static
double const&
f64_inf() {
  static double ret(_f64_inf());
  return ret;
}

static
float const&
f32_ninf() {
  static float ret(_f64_ninf());
  return ret;
}
static
double const&
f64_ninf() {
  static double ret(_f64_ninf());
  return ret;
}

scalar_t scalar_t::make_inf(dtype_t dtype) {
  scalar_t ret { .dtype = dtype };
  if(dtype == dtype_t::f32) {
    ret.as_f32() = f32_inf();
  } else if(dtype == dtype_t::f64) {
    ret.as_f64() = f64_inf();
  } else {
    throw std::runtime_error("should not reach");
  }
  return ret;
}

scalar_t scalar_t::make_negative_inf(dtype_t dtype) {
  scalar_t ret { .dtype = dtype };
  if(dtype == dtype_t::f32) {
    ret.as_f32() = f32_ninf();
  } else if(dtype == dtype_t::f64) {
    ret.as_f64() = f64_ninf();
  } else {
    throw std::runtime_error("should not reach");
  }
  return ret;
}

