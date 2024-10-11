#include "../utils/setup.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "misc.h"

int main() {
  auto [init_pl, fini_pl] = make_pls_matrix_all_reduce(10000, 10000, 4);

  relation_t init_rel = relation_t::make_from_placement(init_pl);

  auto [sol, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

  DOUT(sol);

  solve_naive(sol);

  DOUT(sol);
}
