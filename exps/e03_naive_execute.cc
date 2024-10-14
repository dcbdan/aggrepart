#include "../utils/setup.h"
#include "../utils/args.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "misc.h"
#include "server.h"

#include "../engine/run.h"
#include "../engine/wrap_cuda.h"
#include "../engine/fill.h"

void init_data(server_t& server, graph_t const& graph) {
  for(auto const& [tid, tensor]: graph.tensors) {
    int device = tensor.loc();
    uint64_t size = product(tensor.shape())*dtype_size(graph.dtype);
    server.allocate(tid, device, size);

    if(tensor.type() == graph_t::tt_inn) {
      server.fill(tid, scalar_t::make_one(graph.dtype));
    }
  }
}

map<int, tuple<float, float>> 
tensor_ranges(graph_t const& graph, map<int, mem_t> const& gpu_data) 
{
  vector<float> cpu_data;
  map<int, tuple<float, float>> ret;

  for(auto const& [tid, tensor]: graph.tensors) {
    if(tensor.type() != graph_t::tensor_type_t::tt_out) {
      continue;
    }

    mem_t const& mem = gpu_data.at(tid);
    uint64_t nelem = product(tensor.shape());

    cpu_data.resize(nelem);
    _cuda_set_device(mem.device);
    _cuda_handle_error(cudaMemcpy(
      cpu_data.data(), mem.data, mem.size, 
      cudaMemcpyDeviceToHost));

    float mn = cpu_data[0];
    float mx = cpu_data[0];
    for(uint64_t i = 1; i != nelem; ++i) {
      mn = std::min(mn, cpu_data[i]);
      mx = std::max(mx, cpu_data[i]);
    }

    //DOUT(tid << ": " << vector<float>(cpu_data.begin(), cpu_data.begin() + nelem));

    ret.insert({tid, {mn,mx}});
  }

  return ret;
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

  init_data(server, graph);

  _cuda_sync_all(nlocs);
  {
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
