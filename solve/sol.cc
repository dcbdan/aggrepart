#include "sol.h"

sol_t::sol_t(
  vector<sol_t::info_t> const& fini_state,
  map<int, set<int>> const& init_locs)
  : init_locs(init_locs)
{
  for(auto const& info: fini_state) {
    nodes.push_back(node_t { .fini = info, .inns = {} });
  }
}

void sol_t::naive(int which_node) {
  node_t& node = nodes.at(which_node);
  if(node.is_set()) {
    throw std::runtime_error("naive: this node is set!");
  }

  for(int const& elem: node.elems()) {
    node.inns.push_back(
      which_t::make_input(
        elem,
        get_preferred_input_loc(elem, node.loc())));
  }
}

void sol_t::split(int which_node, vector<sol_t::info_t> const& inns)
{
  node_t& node = nodes.at(which_node);
  if(node.is_set()) {
    throw std::runtime_error("split: this node is set!");
  }

  {
    set<int> all_elems;
    for(info_t const& inn: inns) {
      auto expected_size = all_elems.size() + inn.elems.size();
      set_append(all_elems, inn.elems);
      if(all_elems.size() != expected_size) {
        throw std::runtime_error("split subsets must be disjoint");
      }
    }

    if(!set_equal(node.elems(), all_elems)) {
      throw std::runtime_error("invalid split provided");
    }
  }

  node.inns.reserve(inns.size());
  for(info_t const& inn: inns) {
    node.inns.push_back(append(which_node + 1, inn));
  }
}

void sol_t::chain(int which_node, int inn_loc) {
  node_t& node = nodes.at(which_node);
  if(node.is_set()) {
    throw std::runtime_error("chain: this node is set!");
  }

  if(node.loc() == inn_loc) {
    throw std::runtime_error("chaining to the same loc? seems unusual...");
  }

  info_t inn_info { .elems = node.elems(), .loc = inn_loc };

  int inn_id = find(which_node + 1, inn_info);

  if(inn_id == nodes.size()) {
    throw std::runtime_error("tried to chain, but not already here; try split instead");
  }

  node.inns.push_back(which_t::make_node(inn_id));
}

bool sol_t::is_set() const {
  for(auto const& node: nodes) {
    if(!node.is_set()) {
      return false;
    }
  }
  return true;
}

sol_t::which_t sol_t::append(int start_id, sol_t::info_t const& inn) {
  if(inn.elems.size() == 1) {
    int const& elem = *inn.elems.begin();
    set<int> const& elem_init_locs = init_locs.at(elem);
    if(elem_init_locs.count(inn.loc) > 0) {
      // This is an input node and we already have this guy
      return which_t::make_input(elem, inn.loc);
    }
  }

  {
    int found_id = find(start_id, inn);
    if(found_id < nodes.size()) {
      // We already have this inn
      return which_t::make_node(found_id);
    }
  }

  // We need to solve for this inn
  nodes.push_back(node_t { .fini = inn, .inns = {} });
  return which_t::make_node(nodes.size() - 1);
}

int sol_t::find(int start_id, sol_t::info_t const& info) {
  int id = start_id;
  for(; id != nodes.size(); ++id) {
    node_t const& node = nodes.at(id);
    if(node.fini == info) {
      return id;
    }
  }
  return id;
}

int sol_t::get_preferred_input_loc(int elem, int best_loc) {
  set<int> const& locs = init_locs.at(elem);
  if(locs.count(best_loc) > 0) {
    return best_loc;
  }
  if(locs.size() == 0) {
    throw std::runtime_error("why is this init loc empty?");
  }
  // return the next biggest loc, otherwise return the minimum loc
  for(int const& loc: locs) {
    if(loc > best_loc) {
      return loc;
    }
  }
  return *locs.begin();
}

bool operator==(sol_t::info_t const& lhs, sol_t::info_t const& rhs) {
  return lhs.loc == rhs.loc && set_equal(lhs.elems, rhs.elems);
}

sol_t create_init_sol(
  placement_t const& inn_pl,
  placement_t const& out_pl)
{
  if(!vector_equal(inn_pl.total_shape(), out_pl.total_shape())) {
    throw std::runtime_error("create_init_sol: pls must have same shape");
  }
  if(out_pl.has_partials()) {
    throw std::runtime_error("output placement should not have partials");
  }

  placement_t refi_pl = inn_pl.construct_refinement(out_pl.partition);

  std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
    build_get_refi_index_region(out_pl.partition, refi_pl.partition);

  map<int, set<int>> init_locs;
  {
    vector<set<int>> const& init_locs_vec = refi_pl.locations.get();
    for(int id = 0; id != init_locs_vec.size(); ++id) {
      init_locs.insert({id, init_locs_vec[id]});
    }
  }

  if(out_pl.has_partials()) {
    throw std::runtime_error("output must not have any partials");
  }

  auto out_shape = out_pl.partition.block_shape();
  vector<int> out_bid(out_shape.size());

  vector<int> refi_shape_partial = refi_pl.locations.get_shape();

  vtensor_t<set<int>> all_elems(out_shape);
  do {
    // get the refi region for this out bid
    // set the init refi bid
    hrect_t<int> refi_region = get_refi_index_region(out_bid);
    vector<int> refi_bid = vector_mapfst(refi_region);

    // fill out these_locs for each input partila
    set<int>& these_locs = all_elems.at(out_bid);
    do {
      for(int partial = 0; partial != refi_pl.num_partials(); ++partial) {
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
  for(int idx = 0; idx != all_locs.size(); ++idx) {
    set<int> const& es = all_elems_vec[idx];
    set<int> const& locs = all_locs[idx];
    for(int const& loc: locs) {
      fini_state.push_back(sol_t::info_t {
        .elems = es,
        .loc = loc
      });
    }
  }

  return sol_t(fini_state, init_locs);
}
