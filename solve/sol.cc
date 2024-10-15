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
    if(has_input_elem(elem, inn.loc)) {
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

void solve_naive(sol_t& sol) {
  // Note that sol.nodes grows!
  for(int node_id = 0; node_id != sol.nodes.size(); ++node_id) {
    sol.naive(node_id);
  }
}

optional<string> sol_t::check() const {
  for(int nid = 0; nid != nodes.size(); ++nid) {
    auto const& node = nodes[nid];
    set<int> const& elems = node.elems();

    set<int> es;
    for(auto const& inn: node.inns) {
      if(inn.is_input()) {
        es.insert(inn.elem);
        if(!has_input_elem(inn.elem, inn.loc)) {
          return "missing input element";
        }
      } else {
        auto const& inn_node = nodes.at(inn.node_id);
        set_union_into(es, inn_node.elems());
      }
    }
    if(!set_equal(es, elems)) {
      return "node's inputs do not equal expected set!";
    }
  }

  return std::nullopt;
}

bool sol_t::has_input_elem(int elem, int loc) const {
  auto ie = init_locs.find(elem);
  if(ie == init_locs.end()) {
    return false;
  }
  set<int> const& locs = ie->second;
  return locs.count(loc) > 0;
}

void sol_t::node_t::print(sol_t const& self, std::ostream& out) const {
  if(is_set()) {
    auto f = [&](int i) {
      auto const& which = inns[i];
      if(which.is_input()) {
        out << "in(" << which.elem << ")@" << which.loc;
      } else {
        auto const& node = self.nodes[which.node_id];
        out << node.elems() << "@" << node.loc();
      }
    };

    out << "{";
    f(0);
    for(int i = 1; i != inns.size(); ++i) {
      out << ", ";
      f(i);
    }
    out << "}@" << loc();
  } else {
    out << elems() << "@" << loc();
  }
}

bool operator==(sol_t::info_t const& lhs, sol_t::info_t const& rhs) {
  return lhs.loc == rhs.loc && set_equal(lhs.elems, rhs.elems);
}

std::ostream& operator<<(std::ostream& out, sol_t const& sol) {
  if(sol.nodes.size() == 0) {
    throw std::runtime_error("odd, sol is empty...");
  }
  out << "["; sol.nodes[0].print(sol, out);
  for(int i = 1; i != sol.nodes.size(); ++i) {
    out << ", "; sol.nodes[i].print(sol, out);
  }
  out << "]";

  return out;
}
