#pragma once
#include "../utils/setup.h"

#include "sol.h"

struct tree_state_t {
  using function_cost_t  = std::function<double(sol_t const&)>;
  using function_split_t = std::function<vector<sol_t>(sol_t const&, int)>;

  tree_state_t(
    sol_t const& sol_init,
    function_cost_t f_cost,
    function_split_t f_split);

  // 1. pop from pending
  // 2. call f_split
  // 3. call f_cost
  // 4. append items to pending
  // 5. update best if any of the new solutions are better
  void step();

  struct info_t {
    sol_t sol;
    double cost;
    int start_id;
  };

  priority_queue_least<info_t> pending;

  info_t best;

  function_cost_t  f_cost;
  function_split_t f_split;
};

bool operator<(tree_state_t::info_t const& lhs, tree_state_t::info_t const& rhs);
bool operator>(tree_state_t::info_t const& lhs, tree_state_t::info_t const& rhs);
