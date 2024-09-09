#include "placement.h"

int placement_t::num_partials() const {
  vector<int> block_shape = partition.block_shape();
  vector<int> locs_shape = locations.get_shape();

  if(block_shape.size() == locs_shape.size()) {
    throw std::runtime_error("must have partial dimension in locs");
  }

  if(block_shape.size() + 1 == locs_shape.size()) {
    return locs_shape.back();
  }

  throw std::runtime_error("invalid ranks in placement");
}

set<int> const& placement_t::get_locs(vector<int> index, int partial) const
{
  index.push_back(partial);
  return locations.at(index);
}

set<int> const& placement_t::get_locs(int block, int partial) const
{
  return get_locs(partition.block_to_index(block), partial);
}

