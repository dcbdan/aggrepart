#pragma once
#include "../utils/setup.h"

#include "../repr/graph.h"

struct mem_t {
  void* data;
  uint64_t size;
  int gpu;
};

// Assumption:
// 1. all tensors are zero initialized
// 2. all tensors contains f32s
void run_graph(
  graph_t const& graph,
  map<int, mem_t> const& mems);
