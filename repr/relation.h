#pragma once
#include "../utils/setup.h"
#include "../utils/vtensor.h"

#include "partition.h"
#include "placement.h"

struct relation_t {
  partition_t partition;
  vtensor_t<map<int, int>> locations;

  // give tensor ids of 0,1,...
  static relation_t make_from_placement(placement_t const& pl);

  int num_partials() const;

  bool has_partials() const { return num_partials() > 1; }

  placement_t as_placement() const;

  vector<uint64_t> total_shape() const { return partition.total_shape(); }

  // block_partial_index = block + {partial}
  // elem = block_partial_index as int
  vector<int> elem_to_block(int elem) const;
  int elem_to_partial(int elem) const;

  vector<int> elem_to_index(int elem) const;
  int index_to_elem(vector<int> const& index) const;

  hrect_t<uint64_t> get_region(int elem) const;
};
