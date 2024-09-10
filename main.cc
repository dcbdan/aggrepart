#include "setup.h"

#include "partition.h"
#include "placement.h"
#include "transform.h"
#include "cost.h"

topology_t make_fully_connected_topology(
  int num_devices, uint64_t time_per_byte, uint64_t latency)
{
  topology_t ret;
  for(int src = 0; src != num_devices; ++src) {
  for(int dst = 0; dst != num_devices; ++dst) {
    if(src != dst) {
      ret.insert_wire(src, dst, time_per_byte, latency);
    }
  }}
  return ret;
}

tuple<placement_t, placement_t> make_all_reduce(
  int num_devices,
  vector<uint64_t> shape)
{
  partition_t partition;
  {
    vector<partdim_t> pds;
    for(uint64_t const& d: shape) {
      pds.push_back(partdim_t::singleton(d));
    }
    partition = partition_t { .partdims = pds };
  }

  placement_t src_pl = placement_t::make(partition, num_devices);
  {
    vector<set<int>>& locs = src_pl.locations.get();
    for(int i = 0; i != num_devices; ++i) {
      locs[i].insert(i);
    }
  }

  placement_t dst_pl = placement_t::make(partition, 1);
  {
    set<int>& locs = dst_pl.locations.get()[0];
    for(int i = 0; i != num_devices; ++i) {
      locs.insert(i);
    }
  }

  return {src_pl, dst_pl};
}

int main(int argc, char** argv) {
  // TODO: create several variants of all reduce

  int num_devices = 4;
  uint64_t latency = 1;
  uint64_t time_per_byte = 1;
  topology_t topology =
    make_fully_connected_topology(num_devices, time_per_byte, latency);

  uint64_t ni = 10000;
  uint64_t nj = 10000;

  auto [src_pl, dst_pl] = make_all_reduce(num_devices, {ni, nj});

  {
    transform_t naive_transform = transform_t::make_naive_transform(src_pl, dst_pl);
    if(!transform_t::valid(src_pl, dst_pl, naive_transform)) {
      throw std::runtime_error("should not happen: naive transform is not valid");
    }
    vector<move_t> moves = transform_t::make_moves(src_pl, dst_pl, naive_transform);

    uint64_t total_cost = compute_total_cost(topology, moves);
    DOUT("naive total cost: " << total_cost);
  }
}
