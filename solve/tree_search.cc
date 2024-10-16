#include "tree_search.h"

tree_state_t::tree_state_t(
  sol_t const& sol_init,
  tree_state_t::function_cost_t fc,
  tree_state_t::function_split_t fs)
  : f_cost(fc), f_split(fs)
{
  int id = 0;
  for(; id == sol_init.nodes.size(); ++id) {
    auto const& node = sol_init.nodes[id];
    if(!node.is_set()) {
      break;
    }
  }

  best = info_t {
    .sol = sol_init,
    .cost = f_cost(sol_init),
    .start_id = id
  };

  if(id < sol_init.nodes.size()) {
    pending.push(best);
  }
}

void tree_state_t::step() {
  if(pending.size() == 0) {
    throw std::runtime_error("nothing pending");
  }

  vector<sol_t> next_sols;
  int start_id;
  {
    info_t const& info = pending.top(); 
    next_sols = f_split(info.sol, info.start_id);
    start_id = info.start_id + 1;
    pending.pop();
  }

  for(sol_t const& sol: next_sols) {
    info_t info {
      .sol = sol,
      .cost = f_cost(sol),
      .start_id = start_id 
    };    

    if(info.cost < best.cost) {
      best = info;
    }

    if(start_id < sol.nodes.size()) {
      pending.push(info); 
    }
  }
}

bool operator<(tree_state_t::info_t const& lhs, tree_state_t::info_t const& rhs) {
  return lhs.cost < rhs.cost;
}
bool operator>(tree_state_t::info_t const& lhs, tree_state_t::info_t const& rhs) {
  return lhs.cost > rhs.cost;
}
