#include "placement.h"

int placement_t::num_duplicates() const {
  vector<int> block_shape = partition.block_shape();
  vector<int> locs_shape = locations.get_shape();

  if(block_shape.size() == locs_shape.size()) {
    // This is fine, just assume duplicates is one
    return 1;
  }

  if(block_shape.size() + 1 == locs_shape.size()) {
    return locs_shape.back();
  }

  throw std::runtime_error("invalid ranks in placement");
}

