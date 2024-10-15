#pragma once
#include "../utils/setup.h"

#include "sol.h"

#include "../repr/graph.h"
#include "../repr/relation.h"

struct builder_info_t {
  // the first nodes in sol_t have these tensor ids, corresponding to the
  // output relation
  vector<int> out_tids;

  relation_t const& refi_rel;
  partition_t const& out_part;

  int get_inn_tensor_id(int elem, int loc) const {
    map<int, int> const& loc_to_tensor_id = refi_rel.locations.get().at(elem);
    return loc_to_tensor_id.at(loc);
  }
};

// These functions come in pairs.
// Given `refi_rel`, `out_pl`,
// acquire `(init_sol, builder_info, out_rel)` with `builder_init_sol`.
// After solving `init_sol` to `sol`,
// `builder_create_graph(sol, buidler_info)` returns
// a graph that represents the conversion
// from `refi_rel` to `out_rel`.
//
// Here, builder_info is used for meta data management
tuple<sol_t, builder_info_t, relation_t>
builder_init_sol(
  relation_t const& refi_rel,
  placement_t const& out_pl);
graph_t builder_create_graph(
  sol_t const& solved_sol,
  builder_info_t const& builder_info,
  dtype_t dtype,
  optional<castable_t> maybe_castable);

