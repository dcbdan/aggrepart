#include "../utils/setup.h"
#include "../utils/piper.h"
#include "../solve/exec_list.h"

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

struct z3_inputs_t {
  vector<tuple<int, set<int>>> init_locs;
  vector<tuple<set<int>, set<int>>> fini_info;
  vector<subset_info_t> subset_infos;
  vector<tuple<int, int>> resources;

  // Note: This will divvy up the list of subsets and partitions of subsets
  //       by spliting each dimension in half over and over again. For
  //       building subset_infos in a different way, don't use this function.
  static z3_inputs_t init(
    placement_t const& refi_pl,
    placement_t const& fini_pl,
    int size_multiplier = 1)
  {
    vector<int> refi_shape_partial = refi_pl.locations.get_shape();

    vector<tuple<int, set<int>>> init_info;
    {
      vector<set<int>> const& refi_locs = refi_pl.locations.get();
      for(int elem = 0; elem != refi_locs.size(); ++elem) {
        init_info.emplace_back(elem, refi_locs[elem]);
      }
    }

    auto get_elems = [&](hrect_t<int> const& refi_region) {
      vector<int> refi_bid = vector_mapfst(refi_region);

      set<int> ret;
      do {
        int elem = idxs_to_index(refi_shape_partial, refi_bid);
        ret.insert(elem);
      } while(increment_idxs_region(refi_region, refi_bid));

      return ret;
    };

    vector<tuple<set<int>, set<int>>> fini_info; 
    vector<hrect_t<int>> pending;
    {
      std::function<hrect_t<int>(vector<int> const&)> get_refi_index_region =
        build_get_refi_index_region(fini_pl.partition, refi_pl.partition);
  
      auto out_shape = fini_pl.partition.block_shape();
      vector<int> out_bid(out_shape.size());
  
      do {
        // get the refi region for this out bid
        // set the init refi bid
        hrect_t<int> r = get_refi_index_region(out_bid);
        r.emplace_back(0, refi_pl.num_partials());
        pending.push_back(r);
  
        fini_info.emplace_back(get_elems(r), fini_pl.get_locs(out_bid, 0));
      } while(increment_idxs(out_shape, out_bid));
    }

    // a bit slow, but oh well
    vector<hrect_t<int>> added;
    auto has_done_region = [&](hrect_t<int> const& r) {
      for(auto const& r_: added) {
        if(vector_equal(r, r_)) {
          return true;
        }
      }
      return false;
    };

    vector<subset_info_t> subset_infos;
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
      subset_infos.push_back(info);
    }

    // Add all remaining single sized elements
    {
      set<int> elems_seen;
      for(auto const& info: subset_infos) {
        if(info.get_elems().size() == 1) {
          elems_seen.insert(*info.get_elems().begin());
        }
      }
      int nelems = product(refi_shape_partial);
      for(int elem = 0; elem != nelems; ++elem) {
        if(elems_seen.count(elem) == 0) {
          subset_infos.push_back(subset_info_t::init(set<int>{ elem }, 1));
        }
      }
    }

    for(subset_info_t& info: subset_infos) {
      info.size *= size_multiplier;
    }

    return z3_inputs_t {
      .init_locs      = init_info,
      .fini_info      = fini_info,
      .subset_infos   = subset_infos,
      .resources      = {}
    };
  }

  void add_resource(int src, int dst) {
    if(src == dst) {
      throw std::runtime_error("invalid resource to self");
    }
    resources.emplace_back(src, dst);
  }
};

optional<exec_list_t> 
solve_with_z3(
  vector<tuple<int, set<int>>> const& init_locs,      // elem, locs pairs
  vector<tuple<set<int>, set<int>>> const& fini_info, // elems, locs pairs
  vector<subset_info_t> const& subset_infos,          // 
  vector<tuple<int, int>> const& resources)           // src, dst; may include duplicates
{
  string main_py = "./../z3/src/main.py";
  piper_t piper("python3", vector<string>{ main_py });

  piper.writeline("start-subsets");

  for(auto const& info: subset_infos) {
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
    piper.writeline(msg);
  }
  piper.writeline("stop");

  piper.writeline("start-resources");
  for(auto const& [src, dst]: resources) {
    piper.writeline(write_with_ss(src) + "," + write_with_ss(dst));
  } 
  piper.writeline("stop");

  piper.writeline("start-init");
  for(auto const& [elem, locs]: init_locs) {
    piper.writeline(write_with_ss(elem) + "|" + write_with_ss(locs));
  }
  piper.writeline("stop");

  piper.writeline("start-fini");
  for(auto const& [elems, locs]: fini_info) {
    piper.writeline(write_with_ss(elems) + "|" + write_with_ss(locs) + "");
  }
  piper.writeline("stop");

  piper.writeline("end");

  string maybe_success = piper.readline();
  if(maybe_success == "sat") {
    vector<exec_item_t> exec_list;
    string line;
    while(true) {
      line = piper.readline();
      if(line == "done") {
        break;
      }
      vector<string> words = split_line(line, '|');
      // 0    1     2   3    4          5
      // move|elems|src|dst |start_time|end_time
      // form|elems|loc|time|part0|part1|...
      if(words[0] == "move") {
        exec_list.emplace_back(exec_item_t::move_t {
          .elems      = parse_set<int>(words[1]),
          .src        = parse_with_ss<int>(words[2]),
          .dst        = parse_with_ss<int>(words[3]),
          .start_time = parse_with_ss<int>(words[4]),
          .end_time   = parse_with_ss<int>(words[5])
        });
      } else if(words[0] == "form") {
        vector<set<int>> inns;
        for(int i = 4; i != words.size(); ++i) {
          inns.push_back(parse_set<int>(words[i]));
        }
        exec_list.emplace_back(exec_item_t::form_t {
          .elems      = parse_set<int>(words[1]),
          .loc        = parse_with_ss<int>(words[2]),
          .time       = parse_with_ss<int>(words[3]),
          .inn_elems  = inns
        });
      } else {
        throw std::runtime_error("invalid");
      }
    }

    return exec_list;
  } else {
    return std::nullopt;
  }
}

optional<exec_list_t> 
solve_with_z3(
  z3_inputs_t const& x)
{
  if(x.resources.size() == 0) {
    throw std::runtime_error(
      "Resources are empty. make sure to add resources to z3_inputs...");
  }
  return solve_with_z3(x.init_locs, x.fini_info, x.subset_infos, x.resources);
}
 

