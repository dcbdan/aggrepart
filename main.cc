#include "setup.h"

#include "touch.h"
#include "graph.h"
#include "placement.h"
#include "relation.h"

// What are we trying to do?
//   create a graph that is rows to agg plus repartition
//   using (1) naive algorithm
//         (2) other algorihtms
// What do I know?
// * need partition and placements                              DONE
// * implement functions to help out.. just build it and see

// What about a builder?
// 1. start with input placement and output placement
// 2. add ops to do things
// 3. build the graph in the process

struct builder_t {
  void alloc(string name, placement_t data);
  void alloc_(string name, placement_t data, graph_t::tensor_type_t tt);

  map<string, relation_t> data;

  graph_t graph;
};

builder_t naive_aggrepart(
  string inn_name, placement_t inn_pl,
  string out_name, placement_t out_pl)
{
  if(!vector_equal(inn.total_shape(), out.total_shape())) {
    throw std::runtime_error("total shape does not match!");
  }
  if(out.has_duplicates()) {
    throw std::runtime_error("the output has duplicates and needs to be aggregated!");
  }
  if(inn_pl.has_broadcasted_subtensors()) {
    throw std::runtime_error("Every input sub-tensor must only exist at one location");
  }

  partiton_t const& inn_part = inn_pl.partition;
  partiton_t const& out_part = out_pl.partition;

  builder_t b;

  b.alloc_(inn_name, inn_pl, graph_t::tt_inn);
  b.alloc_(out_name, out_pl, graph_t::tt_out);

  relation_t const& inn_rel = b.data.at(inn_name);
  relation_t const& out_rel = b.data.at(out_name);

  int num_aggs = inn_pl.num_duplicates();

  optional<castable_t> castable = std::nullopt;
  if(num_aggs > 1) {
    castable = castable_t::add
  }

  // For each output block, go and get the inputs
  // and aggregate into the output
  vector<int> out_shape = out_part.block_shape();
  vector<int> out_idx(out_shape.size(), 0);
  do {
    hrect_t<uint64_t> out_region = out_part.get_region(out_idx);
    set<int> const& out_locs = out_pl.locations.at(out_idx);

    hrect_t<int> inn_covering = inn_part.get_covering_blocks(out_region);
    vector<int> inn_idx = vector_mapfst(inn_covering);
    do {
      hrect_t<uint64_t> inn_region = inn_part.get_region(inn_idx);

      touch_t touch = touch_t::intersect(inn_region, out_region, castable);
      touch_t to_subset   = touch.inn_to_write();
      touch_t from_subset = touch.write_to_out();

      for(int i = 0; i != num_agg; ++i) {
        auto const& [loc, inn_tensor] = get_inn_tensor(inn_idx, i);

        int subset = -1;
        for(int const& out_loc: out_locs) {
          int out_tensor = out_rel.at(out_idx, out_loc);
          if(loc == out_loc) {
            b.graph.touch(touch, inn_tensor, out_tensor);
          } else {
            if(subset < 0) {
              subset = b.graph.alloc_(loc, to_subset.out_shape(), graph_t::tt_tmp);
              b.graph.touch(to_subset, inn_tensor, subset);
            }
            // Now move the subset to out loc and touch there
            b.graph.touch_unto(from_subset, subset, out_tensor);
          }
        }
      }
    } while(increment_idxs_region(inn_covering, inn_idx));
  } while(increment_idxs(out_shape, out_idx);

  return b;
}

int main () {

}


