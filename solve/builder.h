#pragma once
#include "../utils/setup.h"

#include "sol.h"

#include "../repr/relation.h"

// These functions come in pairs. There
// Given `refi_rel`, `out_pl`, return `(init_sol, out_rel)` with `builder_init_sol`
// such that after solving `init_sol` to `sol`, `builder_create_graph(sol)` returns
// a graph that represents the conversion from `refi_rel` to `out_rel`.
tuple<sol_t, relation_t> builder_init_sol(
  relation_t const& refi_rel,
  placement_t const& out_pl);
graph_t builder_create_graph(sol_t);

////////sol_t create_init_sol(
////////  placement_t const& inn_pl,
////////  placement_t const& out_pl);
////////
////////tuple<
////////  graph_t,
////////  relation_t,  // refi relation
////////  relation_t>  // out  relation
////////create_graph(
////////  sol_t const& sol,
////////  relation_t init_relation,
////////  partition_t out_partition);
////////
