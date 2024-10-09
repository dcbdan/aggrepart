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

placement_t placement_t::construct_refinement(
  partition_t const& other_part) const
{
  partition_t refi_part = partition_t::intersect(this->partition, other_part);

  std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
    build_get_refi_index_region(this->partition, refi_part);

  int num_ps = num_partials();
  placement_t ret = placement_t::make(refi_part, num_ps);

  vector<int> here_block_shape = this->partition.block_shape();
  vector<int> here_bid(here_block_shape.size(), 0);
  do {
    hrect_t<int> refi_region = get_refi_index_region(here_bid);
    for(int p = 0; p != num_ps; ++p) {
      set<int> const& here_locs = this->get_locs(here_bid, p);
      vector<int> refi_bid = vector_mapfst(refi_region);
      do {
        ret.get_locs(refi_bid, p) = here_locs;
      } while(increment_idxs_region(refi_region, refi_bid));
    }
  } while(increment_idxs(here_block_shape, here_bid));

  return ret;
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

