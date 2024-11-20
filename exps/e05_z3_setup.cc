#include "../utils/setup.h"
#include "../utils/args.h"
#include "../utils/piper.h"

#include "../repr/relation.h"
#include "../repr/graph.h"
#include "../repr/cost.h"

#include "../solve/sol.h"
#include "../solve/builder.h"

#include "problems.h"

struct subset_info_t {
  static subset_info_t init(set<int> const& elems, int size) {
    subset_info_t ret;
    ret.size = size;
    ret.partitions.push_back({ elems });

    return ret;
  }

  set<int> const& get_elems() const { return partitions[0][0]; }

  void add_fineist_partition() {
    set<int> const& elems = get_elems();
    for(auto const& p: partitions) {
      if(p.size() == elems.size()) {
        // ok, we've already added the single element partition
        return;
      }
    }

    vector<set<int>> singleton_elems;
    for(int const& elem: elems) {
      singleton_elems.push_back(set<int>{ elem });
    }
    partitions.push_back(singleton_elems);
  }

  int size;
  vector<vector<set<int>>> partitions;
  // ^ partitions[0] = { elems }
};

int main(int argc, char** argv) {
  dtype_t dtype = dtype_t::f32;
  castable_t castable = castable_t::add;

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

  {
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

  /////////////////////////////////////////////
  // The z3 solver expects:
  //   init info:   list of (elem, locs)
  //   fini info:   list of (elems, locs)
  //   subset info: list of (subset, size, valid partitions of subset)
  //   resources:   list of (src->dst, max_bandwidth)

  // init info:
  //   For each (sub block, partial) in init_rel, give it a letter
  //   Assign the locations from init_rel.locations
  // fini info:
  //   For each block in fini_rel, get the corresponding letters and locations
  // > It looks like init info and fini info can be acquired directly from the
  //   sol_t init object obtained from `builder_init_sol`
  //   (This is a job of `builder_init_sol`--to form init and fini info)

  // All partitions:
  // 1. all singleton elements
  // 2. the singleton partition
  // 3. all binary cuts of the partition,
  //    including binary cuts for the replicate dimension
  // TODO: implement this out
  // Note: all ids in sol_t are just ids into the refi rel!
  // Idea: have a set of pending sets, starting with all hrects in out_rel

  // Collect all finished regions we care about
  // (Here, regions include the partials)

  vector<int> refi_shape_partial = init_rel.locations.get_shape();
  auto get_elems = [&](hrect_t<int> const& refi_region) {
    vector<int> refi_bid = vector_mapfst(refi_region);

    set<int> ret;
    do {
      int elem = idxs_to_index(refi_shape_partial, refi_bid);
      ret.insert(elem);
    } while(increment_idxs_region(refi_region, refi_bid));

    return ret;
  };

  vector<tuple<set<int>, set<int>>> fini_info; // elems, locs pair
  vector<hrect_t<int>> pending;
  {
    std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
      build_get_refi_index_region(fini_pl.partition, init_rel.partition);

    auto out_shape = fini_pl.partition.block_shape();
    vector<int> out_bid(out_shape.size());

    do {
      // get the refi region for this out bid
      // set the init refi bid
      hrect_t<int> r = get_refi_index_region(out_bid);
      r.emplace_back(0, init_rel.num_partials());
      pending.push_back(r);

      fini_info.emplace_back(get_elems(r), fini_pl.get_locs(out_bid, 0));
    } while(increment_idxs(out_shape, out_bid));
  }

  vector<hrect_t<int>> added;
  auto has_done_region = [&](hrect_t<int> const& r) {
    for(auto const& r_: added) {
      if(vector_equal(r, r_)) {
        return true;
      }
    }
    return false;
  };

  vector<subset_info_t> ret;
  while(pending.size() > 0) {
    hrect_t<int> r = pending.back();
    pending.pop_back();

    if(has_done_region(r)) {
      continue;
    }
    added.push_back(r);

    // The size of a region does not include the partials portion
    int size = 1;
    for(int i = 0; i != r.size() - 1; ++i) {
      auto const& [b,e] = r[i];
      size *= (e-b);
    }
    // Note: the size isn't really accurate unless everything is square,
    //       right?
    // TODO

    // Partition each axis
    subset_info_t info = subset_info_t::init(get_elems(r), size);
    for(int i = 0; i != r.size(); ++i) {
      auto const& [b,e] = r[i];
      auto n = e - b;
      if(n > 1) {
        auto m = b + n/2;

        auto rL = r;
        rL[i] = tuple<int, int>(b,m);
        pending.push_back(rL);

        auto rR = r;
        rR[i] = tuple<int, int>(m,e);
        pending.push_back(rR);

        info.partitions.push_back({ get_elems(rL), get_elems(rR) });
      }
    }

    info.add_fineist_partition();
    ret.push_back(info);
  }

  // Add all remaining single sized elements
  set<int> elems_seen;
  for(auto const& info: ret) {
    if(info.get_elems().size() == 1) {
      elems_seen.insert(*info.get_elems().begin());
    }
  }
  int nelems = product(refi_shape_partial);
  for(int elem = 0; elem != nelems; ++elem) {
    if(elems_seen.count(elem) == 0) {
      ret.push_back(subset_info_t::init(set<int>{ elem }, 1));
    }
  }

  DOUT("printing the subset infos");
  for(auto const& info: ret) {
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

  DOUT("");
  DOUT("piper stuff ----------------------");

  string singularity = "/usr/bin/singularity";
  string container = "/home/daniel/Containers/z3.sif";
  string main_py = "/home/daniel/Projects/aggrepart/z3/src/main.py";
  piper_t piper(singularity, vector<string>{ "exec", container, "python3", main_py });

  piper.write("start-subsets\n");
  for(auto const& info: ret) {
    string msg = write_with_ss(info.size) + "|";
    for(auto const& partition: info.partitions) {
      for(auto const& subset: partition) {
        msg += write_with_ss(subset);
        msg += "@";
      }
      msg.resize(msg.size() - 1);
      msg += "|";
    }
    msg.resize(msg.size() - 1);
    msg += "\n";
    piper.write(msg);
  }
  piper.write("stop\n");

  piper.write("start-resources\n");
  auto _write_n_resources = [&](vector<int> const& n) {
    for(int const& src: n) {
    for(int const& dst: n) {
      if(src != dst) {
        piper.write(write_with_ss(src) + "," + write_with_ss(dst) + "\n");
      }
    }}
  };
  _write_n_resources({0,1,2,3});
  _write_n_resources({4,5,6,7});
  for(int i = 0; i != 3; ++i) {
    piper.write(write_with_ss(i) + "," + write_with_ss(i+4) + "\n");
  }
  piper.write("stop\n");

  piper.write("start-init\n");
  for(auto const& [elem, locs]: sol.init_locs) {
    piper.write(write_with_ss(elem) + "|" + write_with_ss(locs) + "\n");
  }
  piper.write("stop\n");

  piper.write("start-fini\n");
  for(auto const& [elems, locs]: fini_info) {
    piper.write(write_with_ss(elems) + "|" + write_with_ss(locs) + "\n");
  }
  piper.write("stop\n");

  piper.write("end\n");

  {
    vector<sol_t::direct_t> z3_solution;
    string line;
    while(true) {
      line = piper.readline();
      if(line == "done") {
        break;
      }
      if(line == "no-sat") {
        throw std::runtime_error("could not solve");
      }

      vector<string> lines = split_line(line, '|');
      // loc|time|elems|inn0|inn1|...
      // 3|0|[4,5,6,7]|[4,5]@3|[6,7]@3

      int loc = parse_with_ss<int>(lines[0]);
      int time = parse_with_ss<int>(lines[1]);
      set<int> elems = parse_set<int>(lines[2]);
      vector<tuple<set<int>, int>> inns;
      for(int idx = 3; idx != lines.size(); ++idx) {
        vector<string> ls = split_line(lines[idx], '@');
        if(ls.size() != 2) {
          throw std::runtime_error("could not split inns into two along '@'");
        }
        inns.emplace_back(
          parse_set<int>(ls.at(0)),
          parse_with_ss<int>(ls.at(1)));
      }
      if(inns.size() > 0) {
        z3_solution.push_back(sol_t::direct_t {
          .elems = elems,
          .loc = loc,
          .time = time,
          .inns = inns
        });
        DOUT(elems << "@" << loc << "|t=" << time);
        for(auto const& [elems, loc]: inns) {
          DOUT("  " << elems << "@" << loc);
        }
      } else {
        // If there are no inns from the z3 solution, it is an input node
      }
    }

    sol_t sol_z3(z3_solution, sol.init_locs);
    DOUT(sol_z3);

    graph_t graph = builder_create_graph(sol_z3, builder_info, dtype, castable);

    std::ofstream f("g_z3.gv");
    graph.print_graphviz(f);
    DOUT("printed g_z3.gv");
  }

}

