#include "setup.h"

#include "partition.h"
#include "placement.h"
#include "transform.h"
#include "cost.h"
#include "ops.h"

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

struct transform_list_t {
  transform_list_t(placement_t const& init_pl)
    : init_pl(init_pl)
  {}

  placement_t init_pl;
  vector<transform_t> transforms;
  vector<placement_t> pls;

  placement_t const& get_last_placement() const {
    if(pls.size() == 0) {
      return init_pl;
    } else {
      return pls.back();
    }
  }

  void insert(transform_t const& t, placement_t const& out) {
    transforms.push_back(t);
    pls.push_back(out);
  }

  struct convert_t {
    placement_t const& inn_pl;
    transform_t const& transform;
    placement_t const& out_pl;
  };

  struct iter_t {
    iter_t(int i, transform_list_t const* self):
      i(i), self(self)
    {}

    iter_t& operator++() {
      i += 1;
      return *this;
    }
    iter_t operator++(int) {
      return iter_t(i+1, self);
    }

    convert_t operator*() const {
      if(i == self->transforms.size()) {
        throw std::runtime_error("this is the end iterator: can't dereference");
      }
      if(i == 0) {
        return convert_t {
          .inn_pl = self->init_pl,
          .transform = self->transforms[0],
          .out_pl = self->pls[0]
        };
      } else {
        return convert_t {
          .inn_pl = self->pls[i-1],
          .transform = self->transforms[i],
          .out_pl = self->pls[i]
        };
      }
    }

    bool equals(iter_t const& other) const {
      if(self != other.self) {
        throw std::runtime_error("can't compare");
      }
      return i == other.i;
    }
  private:
    int i;
    transform_list_t const* self;
  };

  iter_t begin() const {
    return iter_t(0, this);
  }
  iter_t end() const {
    return iter_t(transforms.size(), this);
  }
};
bool operator==(transform_list_t::iter_t const& lhs, transform_list_t::iter_t const& rhs) {
  return lhs.equals(rhs);
}
bool operator!=(transform_list_t::iter_t const& lhs, transform_list_t::iter_t const& rhs) {
  return !(lhs.equals(rhs));
}

int main(int argc, char** argv) {
  // TODO: create several variants of all reduce

  int num_devices = 8;
  uint64_t latency = 1;
  uint64_t time_per_byte = 1;
  topology_t topology =
    make_fully_connected_topology(num_devices, time_per_byte, latency);

  uint64_t ni = 10000;
  uint64_t nj = 10000;

  auto [src_pl, dst_pl] = make_all_reduce(num_devices, {ni, nj});

  //{
  //  transform_t naive_transform = transform_t::make_naive_transform(src_pl, dst_pl);
  //  if(!transform_t::valid(src_pl, dst_pl, naive_transform)) {
  //    throw std::runtime_error("should not happen: naive transform is not valid");
  //  }
  //  vector<move_t> moves = transform_t::make_moves(src_pl, dst_pl, naive_transform);

  //  uint64_t total_cost = compute_total_cost(topology, moves);
  //  DOUT("naive total cost: " << total_cost);
  //}

  {
    transform_list_t list(src_pl);
    while(list.get_last_placement().num_partials() > 1) {
      DOUT("number of partials: " << list.get_last_placement().num_partials());
      auto [transform, next_pl] = reduce_adjacent(
        vector_iota<int>(num_devices),
        list.get_last_placement()).value();
      list.insert(transform, next_pl);
    }

    list.insert(
      transform_t::make_naive_transform(list.get_last_placement(), dst_pl),
      dst_pl);

    uint64_t total_cost = 0;
    for(auto const& [src_pl, transform, dst_pl]: list) {
      vector<move_t> moves = transform_t::make_moves(src_pl, dst_pl, transform);
      total_cost += compute_total_cost(topology, moves);
    }

    DOUT("with reduce adjacent then naive: " << total_cost);
  }
}
