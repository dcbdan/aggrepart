#pragma once
#include "setup.h"

template typename <T>
using hrect_t = vector<tuple<T, T>>;

// This is an error on empty intersection
hrect<uint64_t>
hrect_intersect(
  hrect<uint64_t> const& lhs,
  hrect<uint64_t> const& rhs);

optional<tuple<uint64_t, uint64_t>>
interval_intersect(
  tuple<uint64_t, uint64_t> const& lhs,
  tuple<uint64_t, uint64_t> const& rhs);

