#pragma once
#include "setup.h"

template <typename T>
using hrect_t = vector<tuple<T, T>>;

// This is an error on empty intersection
hrect_t<uint64_t>
hrect_intersect(
  hrect_t<uint64_t> const& lhs,
  hrect_t<uint64_t> const& rhs);

optional<tuple<uint64_t, uint64_t>>
interval_intersect(
  tuple<uint64_t, uint64_t> const& lhs,
  tuple<uint64_t, uint64_t> const& rhs);

bool hrect_has_intersect(
  hrect_t<uint64_t> const& lhs,
  hrect_t<uint64_t> const& rhs);

template <typename T>
void
hrect_union_inplace(
  hrect_t<T>& ret,
  hrect_t<T> const& other)
{
  if(ret.size() != other.size()) {
    throw std::runtime_error("invalid hrect union");
  }
  for(int i = 0; i != ret.size(); ++i) {
    auto& [b,e] = ret[i];
    auto const& [ob, oe] = other[i];
    b = std::min(b, ob);
    e = std::max(e, oe);
  }
}

template <typename T>
T hrect_size(hrect_t<T> const& hrect)
{
  T ret = 1;
  for(auto const& [beg,end]: hrect) {
    ret *= (end-beg);
  }
  return ret;
}

template <typename T>
vector<T> hrect_shape(hrect_t<T> const& hrect) {
  vector<T> ret;
  ret.reserve(hrect.size());
  for(auto const& [beg,end]: hrect) {
    ret.push_back(end-beg);
  }
  return ret;
}
