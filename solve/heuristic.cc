#include "heuristic.h"

void heuristic02(sol_t& sol) {
  using node_t = sol_t::node_t;
  using which_t = sol_t::which_t;

  for(int which_node = 0; which_node != sol.nodes.size(); ++which_node) {
    bool solved = false;
    {
      node_t& node = sol.nodes[which_node];

      if(node.is_set()) {
        // Nothing to do, is already set
        continue;
      }

      for(int i = which_node + 1; i != sol.nodes.size(); ++i) {
        node_t const& next_node = sol.nodes[i];
        if(set_equal(node.elems(), next_node.elems())) {
          node.inns.push_back(which_t::make_node(i));
          solved = true;
          break;
        }
      }
    }

    if(!solved) {
      sol.naive(which_node);
    }
  }
}
