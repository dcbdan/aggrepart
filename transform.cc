#include "transform.h"

bool transform_t::valid(
  placement_t const& inn_pl,
  placement_t const& out_pl,
  transform_t const& transform)
{
  partition_t const& inn_part = inn_pl.partition;
  partition_t const& out_part = out_pl.partition;
  partition_t mid_part = partition_t::intersect(inn_part, out_part);

  vector<int> m_block_shape = mid_part.block_shape();
  vector<int> m_block_partial_shape = m_block_shape;

  int num_inn_partials = inn_pl.num_partials();
  int num_out_partials = out_pl.num_partials();

  m_block_partial_shape.push_back(num_out_partials);

  // mid_info[mid block, out partial] -> out_loc -> inn partials
  vtensor_t<map<int, set<int>>> mid_info(m_block_partial_shape);
  auto mid_info_at = [&](vector<int> const& idx, int out_partial)
    -> map<int, set<int>>&
  {
    vector<int> x = idx;
    x.push_back(out_partial);
    return mid_info.at(x);
  };
  auto mid_info_insert = [&](
    vector<int> const& idx,
    int out_loc,
    int inn_partial,
    int out_partial)
  {
    map<int, set<int>>& info = mid_info_at(idx, out_partial);
    auto [_, did_insert] = info[out_loc].insert(inn_partial);
    return did_insert;
  };

  auto get_mid_index = [&](int inn_block, int out_block) {
    vector<int> inn_idx = inn_part.block_to_index(inn_block);
    vector<int> out_idx = out_part.block_to_index(out_block);
    hrect_t<uint64_t> mid_region = hrect_intersect(
      inn_part.get_region(inn_idx),
      out_part.get_region(out_idx));
    hrect_t<int> ret = mid_part.get_exact_covering_blocks(mid_region);
    if(hrect_size(ret) != 1) {
      throw std::runtime_error("should not occur");
    }
    return vector_mapfst(ret);
  };

  for(auto const& [inn_piece, out_piece]: transform.ops) {
    vector<int> mid_index = get_mid_index(inn_piece.block, out_piece.block);
    bool did_insert = mid_info_insert(
      mid_index, out_piece.loc, inn_piece.partial, out_piece.partial);
    if(!did_insert) {
      // you can't write the same partial to (mid block, out partial, out loc)
      // twice!
      DLINE;
      return false;
    }
  }

  // Make sure that for each block, the partials are valid
  {
    vector<int> mid_idx(m_block_shape.size(), 0);
    do {
      vector<set<int>> inn_partials;
      for(int p = 0; p != num_out_partials; ++p) {
        // dd: out_loc -> inn partials
        map<int, set<int>> dd = mid_info_at(mid_idx, p);
        // dd should contain for the same set of partials for all
        // locations and there should be atleast one location with values

        auto iter = dd.begin();
        if(iter == dd.end()) {
          DLINE;
          // no locations with values
          return false;
        }
        inn_partials.push_back(iter->second);
        iter++;
        for(; iter != dd.end(); ++iter) {
          if(!set_equal(inn_partials[p], iter->second)) {
            print_set(inn_partials[p]); std::cout << std::endl;
            print_set(iter->second);    std::cout << std::endl;
            // In this case, we have the same partial but they have different values
            DLINE;
            return false;
          }
        }
      }

      // Make sure that inn_partials partions the set [0,...,num_inn_partials-1]
      // (make sure that each inn partial is accounted for exactly once)
      map<int, int> counts;
      for(set<int> const& ps: inn_partials) {
        for(int const& p: ps) {
          counts[p]++;
        }
      }
      for(int p = 0; p != num_inn_partials; ++p) {
        if(counts[p] != 1) {
          DLINE;
          return false;
        }
      }
    } while(increment_idxs(m_block_shape, mid_idx));
  }

  // Make sure each (mid block, out partial, out loc) pair is written to exactly once

  auto get_out_index = [&](vector<int> const& mid_index)
  {
    hrect_t<uint64_t> region = mid_part.get_region(mid_index);
    hrect_t<int> ret = out_part.get_exact_covering_blocks(region);
    if(hrect_size(ret) != 1) {
      throw std::runtime_error("should not occur");
    }
    return vector_mapfst(ret);
  };

  {
    vector<int> mid_idx(m_block_shape.size(), 0);
    do {
      for(int p = 0; p != num_out_partials; ++p) {
        set<int> mid_locs;
        for(auto const& [loc, _]: mid_info_at(mid_idx, p)) {
          mid_locs.insert(loc);
        }

        set<int> const& out_locs = out_pl.get_locs(get_out_index(mid_idx), p);

        if(!set_equal(mid_locs, out_locs)) {
          DOUT(mid_locs);
          DOUT(out_locs);
          DLINE;
          return false;
        }
      }
    } while(increment_idxs(m_block_shape, mid_idx));
  }

  // TODO: is that all the checks?
  return true;
}

transform_t transform_t::make_naive_transform(
  placement_t const& inn_pl,
  placement_t const& out_pl)
{
  vector<convert_t> ret;

  partition_t const& inn_part = inn_pl.partition;
  partition_t const& out_part = out_pl.partition;
  partition_t mid_part = partition_t::intersect(inn_part, out_part);

  vector<int> i_shape = inn_part.block_shape();
  vector<int> o_shape = out_part.block_shape();
  vector<int> m_shape = mid_part.block_shape();

  int num_i_partials = inn_pl.num_partials();
  int num_o_partials = out_pl.num_partials();

  if(num_i_partials < num_o_partials) {
    throw std::runtime_error("cannot increase the number of partials");
  }

  vector<int> _to_out_ps;
  {
    vector<int> xs = divide_evenly_int(num_o_partials, num_i_partials);
    for(int o = 0; o != xs.size(); ++o) {
      for(int x = 0; x != xs[o]; ++x) {
        _to_out_ps.push_back(o);
      }
    }
  }
  auto get_out_partial = [&](int which_inn_partial) {
    return _to_out_ps.at(which_inn_partial);
  };

  auto get_block_pair = [&](vector<int> const& m_idx)
    -> tuple<int, int>
  {
    hrect_t<uint64_t> region = mid_part.get_region(m_idx);
    vector<int> inn_idx = inn_part.get_covering_block(region);
    vector<int> out_idx = out_part.get_covering_block(region);
    return {
      inn_part.index_to_block(inn_idx),
      out_part.index_to_block(out_idx)
    };
  };

  map<int, map<int, int>> src_dst_counts;
  auto pick_next_src = [&](set<int> const& src_locs, int dst) {
    if(src_locs.size() == 0) {
      throw std::runtime_error("empty src locs");
    }
    auto iter = src_locs.begin();
    int ret = *iter++;
    int best = src_dst_counts[ret][dst];
    iter++;
    for(; iter != src_locs.end(); ++iter) {
      int cnt = src_dst_counts[*iter][dst];
      if(cnt < best) {
        ret = *iter;
        best = cnt;
      }
    }
    return ret;
  };

  auto broadcast = [&](set<int> const& inn_locs, set<int> const& out_locs)
    -> vector<tuple<int, int>>
  {
    vector<tuple<int, int>> ret;
    for(int const& out: out_locs) {
      if(inn_locs.count(out) == 1) {
        ret.emplace_back(out, out);
      } else {
        ret.emplace_back(pick_next_src(inn_locs, out), out);
      }
    }
    return ret;
  };

  // for each mid block,
  //   for each inn partial,
  //     get the corresponding out partial
  //     broadcast the inn partial to the outputs
  vector<int> m_idx(m_shape.size(), 0);
  do {
    auto [i_idx, o_idx] = get_block_pair(m_idx);
    for(int i_p = 0; i_p != num_i_partials; ++i_p) {
      int o_p = get_out_partial(i_p);
      set<int> const& inn_locs = inn_pl.get_locs(i_idx, i_p);
      set<int> const& out_locs = out_pl.get_locs(o_idx, o_p);
      for(auto const& [i_loc, o_loc]: broadcast(inn_locs, out_locs)) {
        ret.push_back(convert_t {
          .inn = { .block = i_idx, .partial = i_p, .loc = i_loc },
          .out = { .block = o_idx, .partial = o_p, .loc = o_loc }
        });
      }
    }
  } while(increment_idxs(m_shape, m_idx));

  return transform_t {
    .ops = ret
  };
}

vector<move_t> transform_t::make_moves(
  placement_t const& inn_pl,
  placement_t const& out_pl,
  transform_t const& transform)
{
  partition_t const& inn_part = inn_pl.partition;
  partition_t const& out_part = out_pl.partition;

  auto get_size = [&](int inn_block, int out_block) -> uint64_t {
    vector<int> inn_idx = inn_part.block_to_index(inn_block);
    vector<int> out_idx = out_part.block_to_index(out_block);
    hrect_t<uint64_t> mid_region = hrect_intersect(
      inn_part.get_region(inn_idx),
      out_part.get_region(out_idx));
    return hrect_size(mid_region);
  };

  vector<move_t> ret;
  ret.reserve(transform.ops.size());
  for(auto const& [inn_piece, out_piece]: transform.ops) {
    if(inn_piece.loc == out_piece.loc) {
      // no move happens here
      continue;
    }
    // find the subset region and do the move
    ret.push_back(move_t {
      .src = inn_piece.loc,
      .dst = out_piece.loc,
      .size = get_size(inn_piece.block, out_piece.block)
    });
  }

  return ret;
}

