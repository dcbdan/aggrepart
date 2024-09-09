#pragma once
#include "setup.h"

struct move_t {
  int src;
  int dst;
  uint64_t size;
};

struct topology_t {
  void insert_wire(int src, int dst, uint64_t time_per_byte, uint64_t latency = 0);

  struct wire_t {
    int src;
    int dst;
    uint64_t time_per_byte;
    uint64_t latency;
  };

  uint64_t cost(int src, int dst, uint64_t bytes) const;

  vector<wire_t> wires;
  map<int, map<int, int>> src_dst_idx;
};

uint64_t compute_total_cost(
  topology_t const& topology,
  vector<move_t> const& moves);

std::ostream& operator<<(std::ostream& out, move_t const& m);
