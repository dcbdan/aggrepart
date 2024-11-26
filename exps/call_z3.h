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
 

