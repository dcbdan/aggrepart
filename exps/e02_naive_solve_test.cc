#include "../utils/setup.h"
#include "../utils/args.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "misc.h"

int main(int argc, char** argv) {
  args_t args(argc, argv);
  args.set_default<bool>("canonical", true);
  args.set_default<uint64_t>("nrow", 10000);
  args.set_default<uint64_t>("ncol", 10000);

  bool canonical = args.get<bool>("canonical");
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");
  int nlocs = 4;

  auto [init_pl, fini_pl] =
    canonical
    ? make_pls_canonical_4locs_rows_to_cols(nrow, ncol)
    : make_pls_matrix_all_reduce(nrow, ncol, nlocs);

  relation_t init_rel = relation_t::make_from_placement(init_pl);

  auto [sol, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

  DOUT(sol);

  solve_naive(sol);

  DOUT(sol);

  graph_t graph = builder_create_graph(sol, builder_info);

  std::ofstream f("g.gv");
  graph.print_graphviz(f);
  DOUT("printed g.gv");
}
