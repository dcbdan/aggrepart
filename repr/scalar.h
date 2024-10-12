#pragma once
#include "../utils/setup.h"

enum class castable_t {
  add,
  min,
  max
};

enum class dtype_t {
  f32,
  f64
};

uint64_t dtype_size(dtype_t d);

#define _scalar_data_size 16

struct scalar_t {
  dtype_t dtype;
  uint8_t data[_scalar_data_size];

  static scalar_t make_zero(dtype_t dtype);
  static scalar_t make_inf(dtype_t dtype);
  static scalar_t make_negative_inf(dtype_t dtype);

  static scalar_t make_zero(castable_t castable, dtype_t dtype);

  float&  as_f32() { return *reinterpret_cast<float*>(data);  }
  double& as_f64() { return *reinterpret_cast<double*>(data); }
};
