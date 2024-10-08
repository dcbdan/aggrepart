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
    node.inns.push_back(info_t::singleton(
      elem,
      get_preferred_input_loc(elem, node.loc())));
  }
}

void sol_t::split(int which_node, vector<sol_t::info_t> const& inns) {

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

  for(info_t const& inn: inns) {
    append(which_node + 1, inn);
  }
  node.inns = inns;
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

  node.inns.push_back(inn_info);
}

bool sol_t::is_set() const {
  for(auto const& node: nodes) {
    if(!node.is_set()) {
      return false;
    }
  }
  return true;
}

void sol_t::append(int start_id, sol_t::info_t const& inn) {
  if(inn.elems.size() == 1) {
    int const& elem = *inn.elems.begin();
    set<int> const& elem_init_locs = init_locs.at(elem);
    if(elem_init_locs.count(inn.loc) > 0) {
      // This is an input node and we already have this guy
      return;
    }
  }

  if(find(start_id, inn) < nodes.size()) {
    // We already have this inn
    return;
  }

  // We need to solve for this inn
  nodes.push_back(node_t { .fini = inn, .inns = {} });
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
