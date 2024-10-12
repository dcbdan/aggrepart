#pragma once
#include "../utils/setup.h"
#include "../utils/hrect.h"

#include "../repr/scalar.h"

struct touch_t {
  struct dim_t {
    uint64_t d_inn;
    uint64_t d_out;
    uint64_t offset_inn;
    uint64_t offset_out;
    uint64_t size;
  };

  vector<dim_t> dims;
  optional<castable_t> castable;
  dtype_t dtype;

  vector<uint64_t> write_shape() const {
    return vector_from_each_member(dims, uint64_t, size); }
  vector<uint64_t> inn_shape() const {
    return vector_from_each_member(dims, uint64_t, d_inn); }
  vector<uint64_t> out_shape() const {
    return vector_from_each_member(dims, uint64_t, d_out); }

  bool uses_full_inn() const { return vector_equal(inn_shape(), write_shape()); }
  bool uses_full_out() const { return vector_equal(out_shape(), write_shape()); }

  touch_t inn_to_write() const;
  touch_t write_to_out() const;

  bool is_identity() const;

  static touch_t intersect(
    hrect_t<uint64_t> inn_region,
    hrect_t<uint64_t> out_region,
    optional<castable_t> castable,
    dtype_t dtype);
};


