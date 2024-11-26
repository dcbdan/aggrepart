#include "../utils/setup.h"
#include "../utils/args.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"
#include "../solve/heuristic.h"

#include "../engine/run.h"
#include "../engine/wrap_cuda.h"
#include "../engine/fill.h"

#include "problems.h"
#include "server.h"
#include "engine_misc.h"
#include "call_z3.h"

void run_it(
  server_t& server,
  string name,
  dtype_t dtype,
  castable_t castable,
  std::function<void(sol_t&)> solve_it,
  relation_t const& init_rel,
  placement_t const& fini_pl,
  bool print_ranges,
  bool print_graph)
{
  auto [sol, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

  solve_it(sol);

  auto maybe_error_msg = sol.check();
  if(maybe_error_msg) {
    throw std::runtime_error(maybe_error_msg.value());
  }
  if(!sol.is_set()) {
    throw std::runtime_error("did not solve the sol");
  }

  DOUT(sol);

  graph_t graph = builder_create_graph(sol, builder_info, dtype, castable);

  if(graph.num_locations() > server.nlocs()) {
    throw std::runtime_error("graph is using more locations than available");
  }

  if(print_graph) {
    std::ofstream f(name + ".gv");
    graph.print_graphviz(f);
    DOUT("printed " + name + ".gv");
  }

  server.clear();
  init_data_to_ones(server, graph);

  _cuda_sync_all(server.nlocs());
  {
    gremlin_t gremlin("Run Graph");
    run_graph(graph, server.data);
    _cuda_sync_all(server.nlocs());
  }

  if(print_ranges) {
    for(auto const& [tid, vv]: tensor_ranges(graph, server.data)) {
      auto const& tensor = graph.tensors.at(tid);
      auto const& [mn,mx] = vv;
      DOUT(tid << ": " << tensor.type() << ", range (" << mn << ", " << mx << ")");
    }
  }
}

void run_z3(
  server_t& server,
  string name,
  dtype_t dtype,
  castable_t castable,
  relation_t const& init_rel,
  placement_t const& fini_pl,
  vector<tuple<int, int>> const& resources,
  bool print_ranges,
  bool print_graph,
  bool print_exec_list = false)
{
  int size_multiplier = 2;
  z3_inputs_t z3_inputs = z3_inputs_t::init(
    init_rel.as_placement(), 
    fini_pl,
    size_multiplier);
    
  z3_inputs.resources = resources;

  auto maybe_sat = solve_with_z3(z3_inputs);
  if(!bool(maybe_sat)) {
    throw std::runtime_error("could not solve with z3!");
  }
  auto const& exec_list = maybe_sat.value();

  if(print_exec_list) {
    for(auto const& op: exec_list) {
      DOUT(op);
    }
  }

  auto [graph, fini_rel] = builder_create_graph_from_list(
    exec_list,
    init_rel,
    fini_pl,
    dtype, castable);

  if(graph.num_locations() > server.nlocs()) {
    throw std::runtime_error("graph is using more locations than available");
  }

  if(print_graph) {
    std::ofstream f(name + ".gv");
    graph.print_graphviz(f);
    DOUT("printed " + name + ".gv");
  }

  server.clear();
  init_data_to_ones(server, graph);

  _cuda_sync_all(server.nlocs());
  {
    gremlin_t gremlin("Run Graph");
    run_graph(graph, server.data);
    _cuda_sync_all(server.nlocs());
  }

  if(print_ranges) {
    for(auto const& [tid, vv]: tensor_ranges(graph, server.data)) {
      auto const& tensor = graph.tensors.at(tid);
      auto const& [mn,mx] = vv;
      DOUT(tid << ": " << tensor.type() << ", range (" << mn << ", " << mx << ")");
    }
  }
}

int main(int argc, char** argv) {
  dtype_t dtype = dtype_t::f32;
  castable_t castable = castable_t::add;

  args_t args(argc, argv);
  args.set_default<bool>("canonical", true);
  args.set_default<uint64_t>("nrow", 10000);
  args.set_default<uint64_t>("ncol", 10000);
  args.set_default<int>("nlocs", 4);

  uint64_t GB = 1000lu * 1000lu * 1000lu;
  args.set_default<uint64_t>("memsize", 10*GB);

  bool canonical = args.get<bool>("canonical");
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");
  int nlocs = args.get<int>("nlocs");
  uint64_t memsize = args.get<uint64_t>("memsize");

  if(canonical && nlocs != 4) {
    throw std::runtime_error("cononical requires nlocs to be 4");
  }

  auto [init_pl, fini_pl] =
    canonical
    ? make_pls_canonical_4locs_rows_to_cols(nrow, ncol)
    : make_pls_matrix_all_reduce(nrow, ncol, nlocs);

  relation_t init_rel = relation_t::make_from_placement(init_pl);

  server_t server(memsize, nlocs);

  bool print_ranges = true;
  bool print_graph = true;
  auto run_it_ = [&](string name, std::function<void(sol_t&)> solve) {
    run_it(server, name, dtype, castable, solve, init_rel, fini_pl, 
      print_ranges, print_graph);
  };

  auto run_z3_ = [&](string name, string resource_type) {
    vector<tuple<int, int>> resources;
    if(resource_type == "connected") {
      for(int i = 0; i != nlocs; ++i) {
        for(int j = 0; j != nlocs; ++j) {
          if(i != j) {
            resources.emplace_back(i, j);
          }
        }
      }
    } else if(resource_type == "ring") {
      vector<int> loc_order;
      if(nlocs == 8) {
        loc_order = vector<int>{ 0, 2, 3, 1, 5, 7, 6, 4 };
      } else if(nlocs == 4) {
        loc_order = vector<int>{ 0, 2, 3, 1 };
      } else if(nlocs == 2) {
        loc_order = vector<int>{ 0, 1 };
      } else {
        throw std::runtime_error("invalid: must have nlocs 2 4 or 8");
      }
      for(int i = 0; i != loc_order.size(); ++i) {
        int j = (i + 1) % loc_order.size();
        resources.emplace_back(i, j);
        resources.emplace_back(j, i);
      }
    } else if(resource_type == "goofy") {
      auto add_to = [&](int i, int j) {
        if(i < nlocs && j < nlocs) {
          resources.emplace_back(i,j);
        }
      };

      for(int i = 0; i != 4; ++i) {
      for(int j = 0; j != 4; ++j) {
        if(i != j) {
          add_to(i, j);
          add_to(j, i);
          add_to(i + 4, j + 4);
          add_to(j + 4, i + 4);
        }
      }}
      vector<tuple<int, int>> rs {
        {0, 2}, {0, 4}, {0, 4}, {1, 3}, 
        {1, 5}, {1, 5}, {2, 3}, {4, 6}, 
        {5, 7}, {6, 7} 
      };
      for(auto const& [i,j]: rs) {
        add_to(i, j);
        add_to(j, i);
      }
    } else {
      throw std::runtime_error("invalid resource_type given to run_z3_");
    }
    run_z3(server, name, dtype, castable, init_rel, fini_pl, 
      resources, print_ranges, print_graph);
  };

  // With some heurstics and sol objects //////////////////////////
  run_it_("naive_setup",     [](sol_t& sol) { solve_naive(sol); });
  run_it_("naive",           [](sol_t& sol) { solve_naive(sol); });
  run_it_("to_one_spot",     [](sol_t& sol) { heuristic02(sol); });

  if(nlocs == 8) {
    run_it_("ring", [](sol_t& sol) { 
      vector<int> loc_order{ 0, 2, 3, 1, 5, 7, 6, 4 };
      heuristic03_ring(sol, loc_order); 
    });
  } else if(nlocs == 4) {
    run_it_("ring", [](sol_t& sol) { 
      vector<int> loc_order{ 0, 2, 3, 1 }; // 1 to 0 is slow, though
      heuristic03_ring(sol, loc_order); 
    });
  } else if(nlocs == 2) {
    run_it_("ring", [](sol_t& sol) { 
      vector<int> loc_order{ 0, 1 }; // 1 to 0 is slow, though
      heuristic03_ring(sol, loc_order); 
    });
  }

  // With z3 //////////////////////////
  run_z3_("with_z3_connected", "connected");

  if(nlocs == 2 || nlocs == 4 || nlocs == 8) {
    run_z3_("with_z3_ring", "ring");
  }

  run_z3_("with_z3_goofy", "goofy");
}

