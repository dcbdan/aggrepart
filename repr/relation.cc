#include "relation.h"

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

