#pragma once
#include "setup.h"

#include "vtensor.h"
#include "partition.h"

// A plcement contains two components:
// 1. A partition, for how a tensors is partitioned along _output_ axis
// 2. The locations that each sub-tensor must live
// Note that "shape" of locations is one dimension higher to represent the
// duplication of the partition---these are the agg dims.
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

  vector<uint64_t> total_shape() const { return partition.total_shape(); }

  int num_duplicates() const; // TODO

  bool has_duplicates() const { return num_duplicates > 1; }

  set<int> broadcasted_subtensors() const; // TODO

  bool has_broadcasted_subtensors() const { return broadcasted_subtensors.size() > 0; }
};

