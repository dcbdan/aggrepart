#include "relation.h"

relation_t relation_t::make_from_placement(placement_t const& pl) {
  relation_t ret {
    .partition = pl.partition,
    .locations = vtensor_t<map<int, int>>(pl.locations.get_shape())
  };

  int id = 0;
  vector<set<int>> const& xss = pl.locations.get();
  vector<map<int, int>>& yss = ret.locations.get();
  for(int i = 0; i != xss.size(); ++i) {
    set<int> const& xs = xss[i];
    map<int, int>& ys = yss[i];
    for(int const& x: xs) {
      ys.insert({ x, id });
      id++;
    }
  }

  return ret;
}

placement_t relation_t::as_placement() const {
  vtensor_t<set<int>> locs(locations.get_shape());

  vector<set<int>>           & ys = locs.get();
  vector<map<int, int>> const& xs = locations.get();

  for(int i = 0; i != xs.size(); ++i) {
    set<int>& y = ys[i];
    map<int, int> const& x = xs[i];
    for(auto const& [key,_]: x) {
      y.insert(key);
    }
  }

  return placement_t {
    .partition = partition,
    .locations = locs
  };
}

int relation_t::num_partials() const {
  return _num_partials(
    partition.block_shape(), locations.get_shape(), "relation");
}

vector<int> relation_t::elem_to_block(int elem) const {
  vector<int> index = elem_to_index(elem);
  return vector<int>(index.begin(), index.begin() + (index.size() - 1));
}

int relation_t::elem_to_partial(int elem) const {
  vector<int> index = elem_to_index(elem);
  return index.back();
}

vector<int> relation_t::elem_to_index(int elem) const {
  return index_to_idxs(locations.get_shape(), elem);
}

int relation_t::index_to_elem(vector<int> const& index) const {
  return idxs_to_index(locations.get_shape(), index);
}

hrect_t<uint64_t> relation_t::get_region(int elem) const {
  return partition.get_region(elem_to_block(elem));
}

