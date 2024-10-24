#pragma once
#include "../utils/setup.h"

#include "../repr/placement.h"

struct sol_t {
  struct info_t {
    set<int> elems;
    int loc;

    static info_t singleton(int elem, int loc) {
      return info_t { .elems = { elem }, .loc = loc };
    }
  };

  struct which_t {
    int node_id;
    int elem;
    int loc;

    bool is_input() const { return node_id < 0; }
    bool is_node() const { return node_id >= 0; }

    static which_t make_input(int elem, int loc)
    { return which_t { -1, elem, loc }; }

    static which_t make_node(int node_id)
    { return which_t { node_id, 0, 0 }; }
  };

  sol_t();

  sol_t(
    vector<info_t> const& fini_state,
    map<int, set<int>> const& init_locs);

  sol_t(
    vector<info_t> const& fini_state,
    int nlocs,
    map<int, set<int>> const& init_locs);

  sol_t(sol_t const& other);

  struct node_t {
    info_t fini;

    set<int> const& elems() const { return fini.elems; }
    int loc() const { return fini.loc; }

    // either this is empty and the node is unset
    // or the union of the inns equals the outs and
    // the node is set
    vector<which_t> inns; // node_ids

    bool is_set() const {
      return inns.size() > 0;
    }

    void print(sol_t const& self, std::ostream& out) const;
  };

  void naive(int which_node);
  void split(int which_node, vector<info_t> const& subsets);
  void chain(int which_node, int loc);

  bool is_set() const;

  vector<node_t> nodes;

  // elem -> locs
  int nlocs;
  std::shared_ptr<map<int, set<int>>> init_locs; 
  // ^ this should never get modified

  int get_preferred_input_loc(int elem, int best_loc);

  // append an insert node if inn is not an input and the (elems,loc) of inn
  // are not available at or after start_id
  which_t append(int start_id, info_t const& inn);

  // return nodes.size() if not found!
  int find(int start_id, info_t const& info);

  optional<string> check() const;
  bool has_input_elem(int elem, int loc) const;
};

void solve_naive(sol_t& sol);

bool operator==(sol_t::info_t const& lhs, sol_t::info_t const& rhs);

std::ostream& operator<<(std::ostream& out, sol_t const& sol);
