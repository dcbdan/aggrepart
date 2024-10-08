#include "placement.h"

placement_t placement_t::make(partition_t const& partition, int num_partials)
{
  vector<int> block_partial_shape = partition.block_shape();
  block_partial_shape.push_back(num_partials);

  return placement_t {
    .partition = partition,
    .locations = vtensor_t<set<int>>(block_partial_shape)
  };
}

int placement_t::num_partials() const {
  return _num_partials(
    partition.block_shape(), locations.get_shape(), "placement");
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

set<int>& placement_t::get_locs(vector<int> index, int partial)
{
  index.push_back(partial);
  return locations.at(index);
}

set<int>& placement_t::get_locs(int block, int partial)
{
  return get_locs(partition.block_to_index(block), partial);
}

int _num_partials(
  vector<int> const& block_shape,
  vector<int> const& elems_shape,
  string const& msg)
{
  if(block_shape.size() == elems_shape.size()) {
    throw std::runtime_error(msg + ": must have partial dimension in elems");
  }

  if(block_shape.size() + 1 == elems_shape.size()) {
    return elems_shape.back();
  }

  throw std::runtime_error(msg + ": invalid ranks in placement");
}

