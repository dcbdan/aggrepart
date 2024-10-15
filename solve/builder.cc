#include "builder.h"

tuple<sol_t, builder_info_t, relation_t>
builder_init_sol(
  relation_t const& refi_rel,
  placement_t const& out_pl)
{
  if(!vector_equal(refi_rel.total_shape(), out_pl.total_shape())) {
    throw std::runtime_error("create_init_sol: pls must have same shape");
  }
  if(out_pl.has_partials()) {
    throw std::runtime_error("output placement should not have partials");
  }

  std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
    build_get_refi_index_region(out_pl.partition, refi_rel.partition);

  map<int, set<int>> init_locs; // elem -> locations
  {
    vector<map<int,int>> const& to_tensor_id = refi_rel.locations.get();
    for(int elem = 0; elem != to_tensor_id.size(); ++elem) {
      set<int>& elem_locs = init_locs[elem];
      map<int, int> const& elem_loc_tensor_id = to_tensor_id[elem];
      for(auto const& [loc, _]: elem_loc_tensor_id) {
        elem_locs.insert(loc);
      }
    }
  }

  auto out_shape = out_pl.partition.block_shape();
  vector<int> out_bid(out_shape.size());

  vector<int> refi_shape_partial = refi_rel.locations.get_shape();

  vtensor_t<set<int>> all_elems(out_shape);
  do {
    // get the refi region for this out bid
    // set the init refi bid
    hrect_t<int> refi_region = get_refi_index_region(out_bid);
    vector<int> refi_bid = vector_mapfst(refi_region);

    // fill out these_locs for each input partial
    set<int>& these_locs = all_elems.at(out_bid);
    do {
      for(int partial = 0; partial != refi_rel.num_partials(); ++partial) {
        // use this refi_bid, partial to add elems to all elems
        vector<int> elem_vec = vector_concatenate(refi_bid, {partial});
        int elem = idxs_to_index(refi_shape_partial, elem_vec);
        these_locs.insert(elem);
      }
    } while(increment_idxs_region(refi_region, refi_bid));
  } while(increment_idxs(out_shape, out_bid));


  vector<set<int>> const& all_elems_vec = all_elems.get();
  vector<set<int>> const& all_locs      = out_pl.locations.get();
  if(all_locs.size() != all_elems_vec.size()) {
    throw std::runtime_error("must be the same size");
  }

  vector<sol_t::info_t> fini_state;
  int num_out_tensors = 0;
  for(int idx = 0; idx != all_locs.size(); ++idx) {
    set<int> const& es = all_elems_vec[idx];
    set<int> const& locs = all_locs[idx];

    num_out_tensors += locs.size();

    for(int const& loc: locs) {
      fini_state.push_back(sol_t::info_t {
        .elems = es,
        .loc = loc
      });
    }
  }

  int start_tid;
  {
    int max_tid = 0;
    for(map<int, int> const& to_tensor_id: refi_rel.locations.get()) {
      for(auto const& [_, tid]: to_tensor_id) {
        max_tid = std::max(max_tid, tid);
      }
    }
    start_tid = max_tid + 1;
  }

  vector<int> out_tids = vector_iota<int>(num_out_tensors, start_tid);

  relation_t out_rel {
    .partition = out_pl.partition,
    .locations = vtensor_t<map<int, int>>(out_pl.locations.get_shape())
  };

  auto out_tid_iter = out_tids.begin();
  vector<map<int, int>>& out_rel_locs = out_rel.locations.get();
  for(int idx = 0; idx != all_locs.size(); ++idx) {
    map<int, int>& to_tensor_id = out_rel_locs[idx];
    set<int> const& locs = all_locs[idx];

    for(int const& loc: locs) {
      to_tensor_id.insert({loc, *out_tid_iter});
      out_tid_iter++;
    }
  }

  return {
    sol_t(fini_state, init_locs),
    builder_info_t {
      .out_tids = out_tids,
      .refi_rel = refi_rel,
      .out_part = out_pl.partition },
    out_rel
  };
}

graph_t builder_create_graph(
  sol_t const& sol,
  builder_info_t const& info,
  dtype_t dtype,
  optional<castable_t> maybe_castable)
{
  if(!sol.is_set()) {
    throw std::runtime_error("expect set sol; did you forget to solve the sol?");
  }

  relation_t const& refi_rel = info.refi_rel;

  graph_t ret(dtype, maybe_castable);
  {
    // insert the refi_rel into the graph
    vector<map<int, int>> const& refi_rel_locs = refi_rel.locations.get();
    for(int elem = 0; elem != refi_rel_locs.size(); ++elem) {
      vector<uint64_t> shape = hrect_shape(refi_rel.get_region(elem));

      map<int, int> const& to_tensor_id = refi_rel_locs[elem];
      for(auto const& [loc, tensor_id]: to_tensor_id) {
        ret.alloc_(tensor_id, loc, shape, graph_t::tensor_type_t::tt_inn);
      }
    }
  }

  int _next_tid = info.out_tids.back() + 1;
  auto get_new_tid = [&] {
    int ret = _next_tid;
    _next_tid += 1;
    return ret;
  };

  vector<int> node_to_tensor(sol.nodes.size(), -1);
  vector<hrect_t<uint64_t>> node_regions(sol.nodes.size());

  auto get_set_hrect_and_is_overlapping = 
    [&](vector<sol_t::which_t> const& inns) 
      -> tuple<hrect_t<uint64_t>, bool>
  {
    vector<hrect_t<uint64_t>> hrects;
    for(auto const& inn: inns) {
      if(inn.is_input()) {
        hrects.push_back(refi_rel.get_region(inn.elem));
      } else {
        hrects.push_back(node_regions.at(inn.node_id));
      }
    }
    
    hrect_t<uint64_t> ret = hrects[0];
    for(int h = 1; h != hrects.size(); ++h) {
      hrect_t<uint64_t> const& eh = hrects[h];
      for(int i = 0; i != eh.size(); ++i) {
        auto& [b, e] = ret[i];
        auto const& [bb, ee] = eh[i];
        b = std::min(b, bb);
        e = std::max(e, ee);
      }
    }

    for(int i = 0; i != hrects.size()-1; ++i) {
      for(int j = i+1; j != hrects.size(); ++j) {
        if(hrect_has_intersect(hrects[i], hrects[j])) {
          return { ret, true };
        }
      }
    }
    return { ret, false };
  };

  // Step 2:
  //   For each node in reverse order,
  //     insert that node into the graph:
  //       for each inn, move it if necc and then
  //       touch it into this tensor
  //     maintain a map from node_id to tensor
  //   Note: the first builder_info.out_tids.size() are tt_out (!)
  for(int node_id = sol.nodes.size() - 1; node_id >= 0; node_id--) {
    auto node = sol.nodes[node_id];
    // 0. get the hrect of the node and determine if these
    //    touches are copies or updates
    auto [out_region, requires_castable] =
      get_set_hrect_and_is_overlapping(node.inns);
    if(requires_castable && !bool(maybe_castable)) {
      throw std::runtime_error("requires a castable but none provided!");
    }

    // 2. allocate this tensor
    int out_tensor_id;
    {
      vector<uint64_t> shape = hrect_shape(out_region);
      if(node_id < info.out_tids.size()) {
        out_tensor_id = info.out_tids[node_id];
        ret.alloc_(
          out_tensor_id, node.loc(), shape,
          graph_t::tensor_type_t::tt_out);
      } else {
        out_tensor_id = get_new_tid();
        ret.alloc_(
          out_tensor_id, node.loc(), shape,
          graph_t::tensor_type_t::tt_tmp);
      }
    }

    // update meta
    node_to_tensor[node_id] = out_tensor_id;
    node_regions[node_id] = out_region;

    // 3. for each node, touch unto this guy
    for(sol_t::which_t const& which: node.inns) {
      int inn_tensor_id;
      hrect_t<uint64_t> inn_region;
      if(which.is_input()) {
        inn_tensor_id = info.get_inn_tensor_id(which.elem, which.loc);
        inn_region = refi_rel.get_region(which.elem);
      } else {
        inn_tensor_id = node_to_tensor.at(which.node_id);
        inn_region = node_regions.at(which.node_id);
      }

      touch_t touch = touch_t::intersect(inn_region, out_region, std::nullopt, dtype);

      // Only add the castable when it is actually needed. This way,
      // the output tensor will only be initialized when needed
      if(requires_castable) {
        touch.castable = maybe_castable;
      }

      ret.touch_unto(touch, inn_tensor_id, out_tensor_id);
    }
  }

  return ret;
}

