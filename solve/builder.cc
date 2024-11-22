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

      map<int, int> const& to_tensor_id = refi_rel_locs.at(elem);
      for(auto const& [loc, tensor_id]: to_tensor_id) {
        ret.alloc_(tensor_id, loc, shape, graph_t::tensor_type_t::tt_inn);
      }
    }
  }

  // This will prevent us from allocating over any of the
  // output tids
  ret.set_min_tensor_id(info.out_tids.back() + 1);

  auto get_new_tid = [&] {
    return ret.new_tensor_id();
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

  map<int, int> sol_id_to_graph_id;

  // Step 2:
  //   For each node in reverse order,
  //     insert that node into the graph:
  //       for each inn, move it if necc and then
  //       touch it into this tensor
  //     maintain a map from node_id to tensor
  //   Note: the first builder_info.out_tids.size() are tt_out (!)
  for(int node_id = sol.nodes.size() - 1; node_id >= 0; node_id--) {
    auto const& node = sol.nodes.at(node_id);

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

      int graph_id = ret.touch_unto(touch, inn_tensor_id, out_tensor_id);
      sol_id_to_graph_id.insert({node_id, graph_id});
    }
  }

  return ret;
}

struct block_mapping_t {
  int insert_elem(
    vector<int> const& index,
    int elem,
    hrect_t<uint64_t> const& tensor_region)
  {
    hrect_t<int> region;
    for(int const& i: index) {
      region.emplace_back(i, i+1);
    }
    return insert(region, set<int>{ elem }, tensor_region);
  }

  int insert(
    hrect_t<int> const& refi_region,
    set<int> const& elems,
    hrect_t<uint64_t> const& tensor_region)
  {
    data.push_back(datum_t {
      .refi_region = refi_region,
      .elems = elems,
      .tensor_region = tensor_region
    });
    int idx = data.size() - 1;

    hrect_to_idx.insert({ refi_region, idx });
    elems_to_idx.insert({ elems,       idx });

    return idx;
  }

  int get_idx(hrect_t<int> const& h) const {
    return hrect_to_idx.at(h);
  }
  int get_idx(set<int> const& elems) const {
    return elems_to_idx.at(elems);
  }
  optional<int> get_maybe(hrect_t<int> const& h) {
    auto iter = hrect_to_idx.find(h);
    if(iter == hrect_to_idx.end()) {
      return std::nullopt;
    } else {
      return iter->second;
    }
  }
  optional<int> get_maybe(set<int> const& h) {
    auto iter = elems_to_idx.find(h);
    if(iter == elems_to_idx.end()) {
      return std::nullopt;
    } else {
      return iter->second;
    }
  }

  template <typename Key>
  void insert_tensor_at(Key const& key, int tensor_id, int loc) {
    datum_t& datum = data[get_idx(key)];
    auto iter = datum.loc_to_tensor_id.find(loc);
    if(iter == datum.loc_to_tensor_id.end()) {
      datum.loc_to_tensor_id.insert({ loc, tensor_id });
    } else {
      throw std::runtime_error("a tensor already inserted at this loc");
    }
  }

  template <typename Key>
  int get_tensor_id(Key const& key, int loc) const {
    return data[get_idx(key)].loc_to_tensor_id.at(loc);
  }
  template <typename Key>
  hrect_t<uint64_t> get_tensor_region(Key const& key) const {
    return data[get_idx(key)].tensor_region;
  }
  template <typename Key>
  hrect_t<int> get_refi_region(Key const& key) const {
    return data[get_idx(key)].refi_region;
  }

  struct datum_t {
    hrect_t<int> refi_region;
    set<int> elems;
    hrect_t<uint64_t> tensor_region;
    map<int, int> loc_to_tensor_id;

    bool has_aggregation() const {
      auto const& [b,e] = refi_region.back();
      return e-b > 1;
    }
    vector<uint64_t> get_tensor_shape() const {
      return hrect_shape(tensor_region);
    }
  };

  struct compare_hrect_t {
    bool operator()(hrect_t<int> const& lhs, hrect_t<int> const& rhs) const {
      for(int i = 0; i != lhs.size(); ++i) {
        auto const& [lx, ly] = lhs[i];
        auto const& [rx, ry] = rhs[i];
        if(lx < rx) { return true;  } if(lx > rx) { return false; }
        if(ly < ry) { return true;  } if(ly > ry) { return false; }
      }
      return false;
    }
  };
  struct compare_elems_t {
    bool operator()(set<int> const& lhs, set<int> const& rhs) const {
      if(lhs.size() < rhs.size()) {
        return true;
      }
      if(lhs.size() > rhs.size()) {
        return false;
      }
      auto liter = lhs.begin();
      auto riter = rhs.begin();
      for(; liter != lhs.end(); ++liter, ++riter) {
        if(*liter < *riter) { return true;  }
        if(*liter > *riter) { return false; }
      }
      return false;
    }
  };

  vector<datum_t> data;

  map<hrect_t<int>, int, compare_hrect_t> hrect_to_idx;
  map<set<int>, int, compare_elems_t> elems_to_idx;
};

struct exec_list_state_t {
  relation_t const& refi_rel;
  relation_t         out_rel; // this isn't modified after the constructor

  graph_t graph;

  block_mapping_t mapping;

  exec_list_state_t(
    relation_t const& refi_rel,
    placement_t const& out_pl,
    dtype_t dtype,
    optional<castable_t> maybe_castable)
    : refi_rel(refi_rel),
      graph(dtype, maybe_castable)
  {
    // Add the input tensors
    {
      // insert the refi_rel into the graph
      vector<map<int, int>> const& refi_rel_locs = refi_rel.locations.get();
      for(int elem = 0; elem != refi_rel_locs.size(); ++elem) {
        hrect_t<uint64_t> tensor_region = refi_rel.get_region(elem);
        vector<uint64_t> shape = hrect_shape(tensor_region);

        int mapping_idx = mapping.insert_elem(
          refi_rel.elem_to_index(elem),
          elem,
          tensor_region);

        map<int, int> const& to_tensor_id = refi_rel_locs.at(elem);
        for(auto const& [loc, tensor_id]: to_tensor_id) {
          graph.alloc_(tensor_id, loc, shape, graph_t::tensor_type_t::tt_inn);
        }
        mapping.data[mapping_idx].loc_to_tensor_id = to_tensor_id;
      }
    }

    std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
      build_get_refi_index_region(out_pl.partition, refi_rel.partition);

    // Add the output tensors, creating the (output) tids
    {
      out_rel = relation_t {
        .partition = out_pl.partition,
        .locations = vtensor_t<map<int, int>>(out_pl.locations.get_shape())
      };
      int num_partials = refi_rel.num_partials();
      auto out_shape = out_pl.partition.block_shape();
      vector<int> out_bid(out_shape.size());
      do {
        hrect_t<int> refi_region = get_refi_index_region(out_bid);

        hrect_t<int> refi_region_with_all_partials = refi_region;
        refi_region_with_all_partials.emplace_back(0, num_partials);

        // fill out elems (there are more efficient ways...)
        vector<int> refi_bid_with_all = vector_mapfst(refi_region_with_all_partials);
        set<int> elems;
        do {
          int elem = refi_rel.index_to_elem(refi_bid_with_all);
          elems.insert(elem);
        } while(increment_idxs_region(refi_region_with_all_partials, refi_bid_with_all));

        hrect_t<uint64_t> tensor_region = _union_elem_tensor_regions(elems);

        // make sure we have this tensor in the mapping information
        int mapping_index = [&] {
          auto maybe = mapping.get_maybe(elems);
          if(maybe) {
            return maybe.value();
          }
          return mapping.insert(
            refi_region_with_all_partials,
            elems,
            tensor_region);
        }();
        auto& datum = mapping.data[mapping_index];

        set<int> const& locs = out_pl.locations.at(out_bid);
        for(int const& loc: locs) {
          if(datum.loc_to_tensor_id.count(loc) == 0) {
            int tid = graph.alloc_(loc, hrect_shape(tensor_region), graph_t::tt_out);
            datum.loc_to_tensor_id.insert({ loc, tid });
          }
        }

        // Now the mapping has all the tids, and we need all the tids at
        // the relation as well
        map<int, int>& loc_to_tid = out_rel.locations.at(out_bid);
        loc_to_tid = datum.loc_to_tensor_id;
      } while(increment_idxs(out_shape, out_bid));
    }
  }

  hrect_t<uint64_t> _union_elem_tensor_regions(set<int> const& elems) const {
    auto iter = elems.begin();
    hrect_t<uint64_t> ret = mapping.get_tensor_region(set<int>{ *iter++ });
    for(; iter != elems.end(); ++iter) {
      hrect_union_inplace(ret, mapping.get_tensor_region(set<int>{ *iter }));
    }
    return ret;
  }

  hrect_t<int> _union_elem_refi_regions(set<int> const& elems) const {
    auto iter = elems.begin();
    hrect_t<int> ret = mapping.get_refi_region(set<int>{ *iter++ });
    for(; iter != elems.end(); ++iter) {
      hrect_union_inplace(ret, mapping.get_refi_region(set<int>{ *iter }));
    }
    return ret;
  }

  int move(set<int> const& elems, int src, int dst, set<int> deps) {
    auto& d = mapping.data[mapping.get_idx(elems)];

    int src_id = d.loc_to_tensor_id.at(src);

    auto iter = d.loc_to_tensor_id.find(dst);
    int dst_id;
    if(iter == d.loc_to_tensor_id.end()) {
      dst_id = graph.alloc(dst, d.get_tensor_shape());
      d.loc_to_tensor_id.insert({ dst, dst_id });
    } else {
      dst_id = iter->second;
    }

    return graph.move(src_id, dst_id, deps);
  }

  vector<int> form(set<int> const& elems, int loc, vector<set<int>> const& inn_elems) {
    optional<int> maybe = mapping.get_maybe(elems);
    if(!bool(maybe)) {
      auto refi_region = _union_elem_refi_regions(elems);
      auto tensor_region = _union_elem_tensor_regions(elems);
      maybe = mapping.insert(refi_region, elems, tensor_region);
    }

    auto& d_out = mapping.data[maybe.value()];
    int out_id;
    {
      auto iter = d_out.loc_to_tensor_id.find(loc);
      if(iter == d_out.loc_to_tensor_id.end()) {
        out_id = graph.alloc(loc, d_out.get_tensor_shape());
        d_out.loc_to_tensor_id.insert({ loc, out_id });
      } else {
        out_id = iter->second;
      }
    }

    vector<int> ret;
    for(set<int> const& inn_es: inn_elems) {
      auto const& d_inn = mapping.data[mapping.get_idx(inn_es)];
      touch_t touch = d_out.has_aggregation()
        ? touch_t::intersect(
            d_inn.tensor_region, d_out.tensor_region,
            graph.castable.value(), graph.dtype)
        : touch_t::intersect(
            d_inn.tensor_region, d_out.tensor_region,
            std::nullopt, graph.dtype)
        ;
      int const& inn_id = d_inn.loc_to_tensor_id.at(loc);
      int touch_id = graph.touch(touch, inn_id, out_id);
      ret.push_back(touch_id);
    }

    return ret;
  }
};

tuple<graph_t, relation_t>
builder_create_graph_from_list(
  exec_list_t const& exec_list,
  relation_t const& refi_rel,
  placement_t const& out_pl,
  dtype_t dtype,
  optional<castable_t> maybe_castable)
{
  // Here we have the following equivalences
  // (1) an hrect (including partials range)
  // (2) elems
  // (3) list of (loc, tensor id)
  // (4) tensor region

  exec_list_state_t state(refi_rel, out_pl, dtype, maybe_castable);
  // exec_list_state_t will
  // 0. add the input tensors
  // 1. add the output tensors
  // into a graph object for us,
  // all will maintaining mappings between hrect, elems, etc

  // For every (src,dst), record, in end time order, all move ids
  map<int, map<int, map<int, vector<int>>>> move_times;
  auto insert_move_time = [&](int src, int dst, int end_time, int move_id) {
    move_times[src][dst][end_time].push_back(move_id);
  };
  auto get_move_time_deps = [&](int src, int dst, int start_time) {
    map<int, vector<int>> const& m = move_times[src][dst];
    if(m.size() == 0) {
      return set<int>();
    }

    auto iter = m.begin();
    auto ret = iter++;
    {
      auto const& [done_time, move_ids] = *ret;
      if(done_time > start_time) {
        return set<int>();
      }
    }
    for(; iter != m.end(); ++iter) {
      auto const& [done_time, move_ids] = *iter;
      if(done_time <= start_time) {
        ret = iter;
      } else {
        break;
      }
    }
    vector<int> const& move_ids = ret->second;
    return set<int>(move_ids.begin(), move_ids.end());
  };

  for(auto const& op: exec_list) {
    if(op.is_move()) {
      auto const& move = op.get_move();
      set<int> deps = get_move_time_deps(move.src, move.dst, move.start_time);
      int move_id = state.move(move.elems, move.src, move.dst, deps);
      insert_move_time(move.src, move.dst, move.end_time, move_id);
    } else if(op.is_form()) {
      auto const& form = op.get_form();
      state.form(form.elems, form.loc, form.inn_elems);
    } else {
      throw std::runtime_error("should not reach");
    }
  }

  return { state.graph, state.out_rel };
}
