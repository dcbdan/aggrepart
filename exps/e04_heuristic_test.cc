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
}

