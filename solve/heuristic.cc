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

void heuristic03_ring(sol_t& sol, vector<int> loc_order) {
  using node_t  = sol_t::node_t;
  using which_t = sol_t::which_t;
  using info_t  = sol_t::info_t;

  map<int, int> prev_loc;
  for(int i = 0; i != loc_order.size(); ++i) {
    int here = i;
    int next = (i+1) % loc_order.size();
    prev_loc[loc_order[next]] = loc_order[here];
  }

  for(int which_node = 0; which_node != sol.nodes.size(); ++which_node) {
    set<int> to_recv;
    int loc;
    {
      node_t& node = sol.nodes[which_node];
      loc = node.loc();

      if(node.is_set()) {
        // Nothing to do, is already set
        continue;
      }

      for(int const& elem: node.elems()) {
        if(sol.has_input_elem(elem, loc)) {
          node.inns.push_back(which_t::make_input(elem, loc));
        } else {
          to_recv.insert(elem);
        }
      }
    }

    if(to_recv.size() > 0) {
      info_t info { 
        .elems = to_recv,
        .loc = prev_loc.at(loc) 
      };

      which_t next_inn = sol.append(which_node + 1, info);

      // NOTE: sol_t::append may append to nodes, so we access
      //       this node only after the append has occured
      node_t& node = sol.nodes[which_node];

      node.inns.push_back(next_inn);
    }
  }
}
