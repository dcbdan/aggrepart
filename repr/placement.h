#pragma once
#include "../utils/setup.h"
#include "../utils/vtensor.h"

#include "partition.h"

// A plcement contains two components:
// 1. A partition, for how a tensors is partitioned along _output_ axis
// 2. The locations that each sub-tensor must live
// Note that "shape" of locations is one dimension higher to represent the
// partials that need to be aggregated together.
//
// Example:
//   partition blocks is 3 x 4
//   locations shape  is 3 x 4 x 1
//   >> Then no aggregation is needed
//
// Example:
//   partition blocks is 3 x 4
//   locations shape  is 3 x 4 x 2
//   >> Then an aggregation is needed
struct placement_t {
  partition_t partition;
  vtensor_t<set<int>> locations;

  static placement_t make(partition_t const& partition, int num_partials);

  vector<uint64_t> total_shape() const { return partition.total_shape(); }

  int num_partials() const;

  bool has_partials() const { return num_partials() > 1; }

  set<int> broadcasted_subtensors() const;

  bool has_broadcasted_subtensors() const { return broadcasted_subtensors().size() > 0; }

  set<int> const& get_locs(vector<int> index, int partial) const;
  set<int> const& get_locs(int block, int partial) const;

  set<int>& get_locs(vector<int> index, int partial);
  set<int>& get_locs(int block, int partial);
};

int _num_partials(
  vector<int> const& block_shape,
  vector<int> const& elems_shape,
  string const& error_header);
