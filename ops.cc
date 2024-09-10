#include "ops.h"

// 1. sort the locations
// 2. group the pairs and send em over
vector<tuple<int, int>>
solve_reduce(
  vector<int> device_order,
  set<int> locs_)
{
  vector<int> locs(locs_.begin(), locs_.end());

  if(locs.size() == 0) {
    throw std::runtime_error("invalid: empty");
  }
  if(locs.size() == 1) {
    return vector<tuple<int, int>>{ {locs[0], locs[0]} };
  }

  auto find = [&](int x) {
    auto iter = std::find(device_order.begin(), device_order.end(), x);
    if(iter == device_order.end()) {
      throw std::runtime_error("loc not in device order");
    }
    return std::distance(device_order.begin(), iter);
  };
  std::function<bool(int, int)> compare = [&](int lhs, int rhs) {
    return find(lhs) < find(rhs);
  };
  std::sort(locs.begin(), locs.end(), compare);

  vector<tuple<int, int>> ret;
  int idx = 0;
  while(idx != locs.size()) {
    ret.emplace_back(locs[idx], locs[idx]);
    if(idx + 1 < locs.size()) {
      ret.emplace_back(locs[idx+1], locs[idx]);
      idx++;
    }
    idx++;
  }
  return ret;
}

optional<tuple<transform_t, placement_t>>
reduce_adjacent(
  vector<int> device_order,
  placement_t const& inn_pl)
{
  int num_inn_partials = inn_pl.num_partials();
  if(num_inn_partials == 1) {
    // nothing to reduce!
    return std::nullopt;
  }

  vector<int> block_shape = inn_pl.partition.block_shape();

  vector<int> block_partial_shape = block_shape;
  block_partial_shape.push_back(num_inn_partials);

  // 1. verify that all partials are at one location:
  {
    vector<int> idx(block_partial_shape.size());
    do {
      set<int> const& locs = inn_pl.locations.at(idx);
      if(locs.size() > 1) {
        return std::nullopt;
      }
      if(locs.size() == 0) {
        throw std::runtime_error("invalid inn_pl argument");
      }
    } while(increment_idxs(block_partial_shape, idx));
  }

  auto get_inn_loc = [&](int block, int partial) {
    set<int> const& ls = inn_pl.get_locs(block, partial);
    return *ls.begin();
  };

  int num_inn_blocks = product(block_shape);

  // 2. verify that any pair of partials within a block
  //    are at different locations
  for(int block = 0; block != num_inn_blocks; ++block) {
    set<int> ls;
    for(int p = 0; p != num_inn_partials; ++p) {
      int loc = get_inn_loc(block, p);
      ls.insert(loc);
    }
    if(ls.size() != num_inn_partials) {
      return std::nullopt;
    }
  }

  transform_t transform;
  vector<transform_t::convert_t>& ops = transform.ops;
  for(int block = 0; block != num_inn_blocks; ++block) {
    // here we have a set of locs with partials and we need to add them together
    map<int, int> loc_to_partial;
    set<int> locs;
    for(int p = 0; p != num_inn_partials; ++p) {
      int loc = get_inn_loc(block, p);
      loc_to_partial.insert({loc, p});
      locs.insert(loc);
    }

    map<int, int> dst_to_dst_p;
    auto get_dst_p = [&](int dst) {
      auto iter = dst_to_dst_p.find(dst);
      if(iter == dst_to_dst_p.end()) {
        auto [new_iter, _] = dst_to_dst_p.insert({dst, dst_to_dst_p.size()});
        iter = new_iter;
      }
      return iter->second;
    };

    for(auto const& [src, dst]: solve_reduce(device_order, locs)) {
      int src_p = loc_to_partial.at(src);
      int dst_p = get_dst_p(dst);

      ops.push_back(transform_t::convert_t {
        .inn = transform_t::piece_t { .block = block, .partial = src_p, .loc = src },
        .out = transform_t::piece_t { .block = block, .partial = dst_p, .loc = dst }
      });
      DOUT(ops.back());
    }
  }

  auto maybe = transform_t::make_placement(inn_pl, inn_pl.partition, transform);
  if(!maybe) {
    // this should cover the case where a pair of blocks end up with a different
    // number of partials
    return std::nullopt;
  }

  placement_t const& out_pl = maybe.value();
  return tuple<transform_t, placement_t> {
    transform,
    out_pl
  };
}

