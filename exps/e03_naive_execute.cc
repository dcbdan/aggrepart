#include "../utils/setup.h"
#include "../utils/args.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "../engine/run.h"
#include "../engine/wrap_cuda.h"
#include "../engine/fill.h"

#include "problems.h"
#include "server.h"
#include "engine_misc.h"

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

  auto [sol, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

  DOUT(sol);

  solve_naive(sol);

  DOUT(sol);

  graph_t graph = builder_create_graph(sol, builder_info, dtype, castable);
  if(graph.num_locations() > nlocs) {
    throw std::runtime_error("graph is using more locations than available");
  }

  server_t server(memsize, nlocs);

  std::ofstream f("g.gv");
  graph.print_graphviz(f);
  DOUT("printed g.gv");

  init_data_to_ones(server, graph);

  for(int i = 0; i != 3; ++i) {
    _cuda_sync_all(nlocs);
    gremlin_t gremlin("Run Graph");
    run_graph(graph, server.data);
    _cuda_sync_all(nlocs);
  }

  for(auto const& [tid, vv]: tensor_ranges(graph, server.data)) {
    auto const& tensor = graph.tensors.at(tid);
    auto const& [mn,mx] = vv;
    DOUT(tid << ": " << tensor.type() << ", range (" << mn << ", " << mx << ")");
  }
}
