#include "../utils/setup.h"
#include "../utils/args.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"
#include "../solve/tree_search.h"

#include "../engine/run.h"
#include "../engine/wrap_cuda.h"
#include "../engine/fill.h"

#include "problems.h"
#include "server.h"
#include "engine_misc.h"

double time_difference_micro_seconds(
  timestamp_t const& beg, 
  timestamp_t const& end)
{
  using namespace std::chrono;
  return double(
    duration_cast<microseconds>(end-beg).count());
}

#include <thread>
void sleep_some() {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(75ms);
}

struct ring_t {
  ring_t() {
    order = vector<int>{ 0, 2, 3, 1, 5, 7, 6, 4 };
  }

  int get_lhs_neighbor(int loc) {
    for(int i = 0; i != order.size(); ++i) {
      if(order[i] == loc) {
        if(i == 0) {
          return order.back();
        } else {
          return order[i-1];
        }
      }
    }
    throw std::runtime_error("should not reach");
  }
  int get_rhs_neighbor(int loc) {
    for(int i = 0; i != order.size(); ++i) {
      if(order[i] == loc) {
        if(i == order.size() - 1) {
          return order[0];
        } else {
          return order[i+1];
        }
      }
    }
    throw std::runtime_error("should not reach");
  }


  vector<int> order;
};

int main(int argc, char** argv) {
  dtype_t dtype = dtype_t::f32;
  castable_t castable = castable_t::add;

  args_t args(argc, argv);
  args.set_default<string>("plan", "allreduce");
  args.set_default<uint64_t>("nrow", 10000);
  args.set_default<uint64_t>("ncol", 10000);
  args.set_default<int>("nlocs", 4);
  args.set_default<bool>("use_ring", false);

  uint64_t GB = 1000lu * 1000lu * 1000lu;
  args.set_default<uint64_t>("memsize", 10);

  string plan = args.get<string>("plan");
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");
  int nlocs = args.get<int>("nlocs");
  uint64_t memsize = GB*args.get<uint64_t>("memsize");

  if(plan == "canonical" && nlocs != 4) {
    throw std::runtime_error("cononical requires nlocs to be 4");
  }

  bool use_ring = args.get<bool>("use_ring");
  if(use_ring && nlocs != 8) {
    throw std::runtime_error("can only use ring with 8 locs");
  }

  auto [init_pl, fini_pl] = [&] {
    if(plan == "canonical") {
      return make_pls_canonical_4locs_rows_to_cols(nrow, ncol);
    } 
    if(plan == "allreduce") {
      return make_pls_matrix_all_reduce(nrow, ncol, nlocs);
    } 
    if(plan == "to-strip") {
      return make_pls_row_strip_to_col_strip(nrow, ncol, nlocs);
    }
    throw std::runtime_error("invalid plan");
  }();

  relation_t init_rel = relation_t::make_from_placement(init_pl);

  server_t server(memsize, nlocs);

  auto [sol_init, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

  // Ugh, not doing a good job having solution objects know anything
  // about the underlying cluster
  sol_init.nlocs = nlocs;

  ring_t ring;
  tree_state_t::function_cost_t f_cost = [&](sol_t const& sol_)
    -> double
  {
    sleep_some();

    sol_t sol = sol_;
    solve_naive(sol);
    graph_t graph = builder_create_graph(sol, builder_info, dtype, castable);

    server.clear();

    init_data_to_ones(server, graph);

    _cuda_sync_all(server.nlocs());
    auto time_start = clock_now();

    run_graph(graph, server.data);

    _cuda_sync_all(server.nlocs());
    auto time_end = clock_now();

    double ms = time_difference_micro_seconds(time_start, time_end);
    DOUT("               cost: " << ms);
    return ms; 
  };

  tree_state_t::function_split_t f_split = [&](sol_t const& sol_init, int start_id) 
    -> vector<sol_t>
  {
    using info_t  = sol_t::info_t;
    using node_t  = sol_t::node_t;
    using which_t = sol_t::which_t;

    node_t const& init_node = sol_init.nodes[start_id];
    if(init_node.is_set()) {
      return { sol_init };
    }

    vector<vector<info_t>> ret;

    // If we can chain, add the chaining as a solution
    for(int id = start_id + 1; id != sol_init.nodes.size(); ++id) {
      node_t const& next_node = sol_init.nodes[id];
      if(init_node.elems() == next_node.elems()) {
        ret.push_back({ 
          info_t { .elems = next_node.elems(), .loc = next_node.loc() } 
        });
      }
    }

    // If we can pop single elements,
    //   pop them and recv the rest from a random other spot
    {
      vector<int> avail;
      set<int> the_rest;
      for(int const& elem: init_node.elems()) {
        if(sol_init.has_input_elem(elem, init_node.loc())) {
          avail.push_back(elem);       
        } else {
          the_rest.insert(elem);
        }
      }

      if(avail.size() > 0) {
        vector<int> other_locs;
        if(use_ring) {
          other_locs.push_back(ring.get_lhs_neighbor(init_node.loc()));
          other_locs.push_back(ring.get_rhs_neighbor(init_node.loc()));
        } else {
          int another_loc = runif(sol_init.nlocs - 1);
          if(another_loc >= init_node.loc()) {
            another_loc++;
          }
        }

        for(int const& another_loc: other_locs) {
          ret.emplace_back();
          vector<info_t>& is = ret.back();
          for(int const& elem: avail) {
            is.push_back(info_t::singleton(elem, init_node.loc()));
          }

          is.push_back(info_t { 
            .elems = the_rest,
            .loc = another_loc
          });
        }
      }
    }

    vector<sol_t> ret_sols;
    for(vector<info_t> const& infos: ret) {
      ret_sols.emplace_back(sol_init);
      sol_t& sol = ret_sols.back();
      vector<which_t> new_inns;
      for(auto const& info: infos) {
        new_inns.push_back(sol.append(start_id+1, info));
      }
      sol.nodes[start_id].inns = new_inns;
    }

    return ret_sols;
  };

  // Run f_cost once to "warm" things up
  f_cost(sol_init);
 
  tree_state_t state(sol_init, f_cost, f_split);
  DOUT("naive solution: " << state.best.cost);

  args.set_default<int>("niter", 100);
  int niter = args.get<int>("niter");

  for(int i = 0; i != niter && state.pending.size() > 0; ++i) {
    state.step();
    DOUT("best solution: " << state.best.cost);
  };
}
