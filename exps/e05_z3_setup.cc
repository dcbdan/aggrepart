#include "../utils/setup.h"
#include "../utils/args.h"
#include "../utils/piper.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "call_z3.h"
#include "problems.h"

int main(int argc, char** argv) {
  dtype_t dtype = dtype_t::f32;
  castable_t castable = castable_t::add;

  args_t args(argc, argv);
  args.set_default<bool>("canonical", true);
  args.set_default<uint64_t>("nrow", 10000);
  args.set_default<uint64_t>("ncol", 10000);
  args.set_default<int>("nlocs", 8);

  bool canonical = args.get<bool>("canonical");
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");
  int nlocs = args.get<int>("nlocs");

  auto [init_pl, fini_pl] =
    canonical
    ? make_pls_canonical_4locs_rows_to_cols(nrow, ncol)
    : make_pls_matrix_all_reduce(nrow, ncol, nlocs);

  relation_t init_rel = relation_t::make_from_placement(init_pl);

  {
    auto [sol, builder_info, fini_rel] = builder_init_sol(init_rel, fini_pl);

    auto sol_naive = sol;
    DOUT(sol_naive);

    solve_naive(sol_naive);

    DOUT(sol_naive);

    graph_t graph = builder_create_graph(sol_naive, builder_info, dtype, castable);

    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("printed g.gv");
  }

  DOUT("\n");

  z3_inputs_t z3_inputs = z3_inputs_t::init(init_pl, fini_pl);

  auto _add_resource_group = [&](vector<int> const& n) {
    for(int const& src: n) {
    for(int const& dst: n) {
      if(src != dst) {
        z3_inputs.add_resource(src, dst);
      }
    }}
  };
  for(int i = 0; i != 8; ++i) {
    z3_inputs.add_resource(i, (i + 1) % 8);
  }

  DOUT("printing the subset infos");
  for(auto const& info: z3_inputs.subset_infos) {
    DOUT(info.get_elems() << " with size " << info.size);
    for(auto const& partition: info.partitions) {
      if(partition.size() > 1) {
        std::cout << "    ";
        for(auto const& subset: partition) {
          std::cout << subset << " ";
        }
        DOUT("");
      }
    }
  }

  auto maybe_sat = solve_with_z3(z3_inputs);
  if(maybe_sat) {
    auto const& exec_list = maybe_sat.value();

    DOUT("//////////////////////////////////////////////////");
    for(auto const& op: exec_list) {
      DOUT(op);
    }
    DOUT("//////////////////////////////////////////////////");

    auto [graph, fini_rel] = builder_create_graph_from_list(
      exec_list,
      init_rel,
      fini_pl,
      dtype, castable);
    DOUT("INIT REL");
    init_rel.print_lines(std::cout);
    DOUT("FINI REL");
    fini_rel.print_lines(std::cout);

    std::ofstream f("g_z3.gv");
    graph.print_graphviz(f);
    DOUT("printed g_z3.gv");
  }
}
