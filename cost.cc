#include "cost.h"

void topology_t::insert_wire(
  int src, int dst, uint64_t time_per_byte, uint64_t latency)
{
  wires.push_back(wire_t {
    .src = src,
    .dst = dst,
    .time_per_byte = time_per_byte,
    .latency = latency
  });
  int idx = wires.size() - 1;
  map<int, int>& m = src_dst_idx[src];
  auto [_, did_insert] = m.insert({dst, idx});
  if(!did_insert) {
    throw std::runtime_error("only one wire allowed between each src, dst pair");
  }
}

uint64_t topology_t::cost(int src, int dst, uint64_t bytes) const {
  auto const& [_0, _1, time_per_byte, latency] =
    wires[src_dst_idx.at(src).at(dst)];
  return latency + time_per_byte*bytes;
}

uint64_t compute_total_cost(
  topology_t const& topology,
  vector<move_t> const& moves)
{
  map<int, map<int, uint64_t>> costs;
  for(auto const& [src, dst, size]: moves) {
    costs[src][dst] += topology.cost(src, dst, size);
  }

  uint64_t ret = 0;
  for(auto const& [_src, m]: costs) {
    for(auto const& [_dst, cost]: m) {
      ret = std::max(ret, cost);
    }
  }

  return ret;
}

std::ostream& operator<<(std::ostream& out, move_t const& m) {
  out << "move{" << m.src << "->" << m.dst << "|" << m.size << "}";
  return out;
}
