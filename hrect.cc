#include "hrect.h"

vector<tuple<uint64_t, uint64_t>>
hrect_intersect(
  vector<tuple<uint64_t, uint64_t>> const& lhs,
  vector<tuple<uint64_t, uint64_t>> const& rhs)
{
  if(lhs.size() != rhs.size()) {
    throw std::runtime_error("hrect_intersect: incorrect sizes");
  }

  vector<tuple<uint64_t, uint64_t>> ret;
  ret.reserve(lhs.size());
  for(int i = 0; i != lhs.size(); ++i) {
    auto const& [lb,le] = lhs[i];
    auto const& [rb,re] = rhs[i];

    uint64_t b = std::max(lb,rb);
    uint64_t e = std::min(le,re);

    if(b >= e) {
      throw std::runtime_error("empty intersection");
    }
    ret.emplace_back(b,e);
  }

  return ret;
}

optional<tuple<uint64_t, uint64_t>>
interval_intersect(
  tuple<uint64_t, uint64_t> const& lhs,
  tuple<uint64_t, uint64_t> const& rhs)
{
  auto const& [lb,le] = lhs;
  auto const& [rb,re] = rhs;

  uint64_t b = std::max(lb,rb);
  uint64_t e = std::min(le,re);

  if(b < e) {
    return tuple<uint64_t, uint64_t>{b, e};
  } else {
    return std::nullopt;
  }
}


