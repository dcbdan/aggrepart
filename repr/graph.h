#pragma once
#include "../utils/setup.h"

#include "touch.h"
#include "scalar.h"

struct graph_t {
  graph_t(
    dtype_t d = dtype_t::f32,
    optional<castable_t> c = std::nullopt);

  enum tensor_type_t {
    tt_inn,
    tt_out,
    tt_tmp
  };

  struct tensor_t {
    tensor_t();

    tensor_t(int loc, vector<uint64_t> shape, tensor_type_t type);

    int loc() const { return _loc; }

    vector<uint64_t> const& shape() const { return _shape; }
    int rank() const { return _shape.size(); }

    tensor_type_t type() const { return _type; }

    void write_with(); // aka: do_a_read_of_this_tensor

    void insert_write(int gid);

    void insert_init(int gid);

    bool is_read_only() const { return read_only; }

    set<int> writes() const { return _writes; }

    set<int> inits() const { return _inits; }

    bool has_init() const { return inits().size() > 0; }
  private:
    int _loc;
    vector<uint64_t> _shape;
    tensor_type_t _type;
    bool read_only;

    // all graph ids that have writen to this tensor
    set<int> _writes;
    set<int> _inits;
    // all graph ids that initialize this tensor
  };

  struct move_t {
    int src_loc;
    int dst_loc;
    uint64_t elem; 
  };

  struct fill_t {
    scalar_t scalar;
    uint64_t elem;
  };

  struct node_t {
    std::variant<touch_t, move_t, fill_t> op;
    int inn_tensor_id; // note used when fill_t
    int out_tensor_id;

    bool is_touch() const { return std::holds_alternative<touch_t>(op); }
    bool is_move()  const { return std::holds_alternative<move_t>(op);  }
    bool is_fill()  const { return std::holds_alternative<fill_t>(op);  }

    touch_t const& get_touch() const { return std::get<touch_t>(op); }
    move_t  const& get_move()  const { return std::get<move_t>(op);  }
    fill_t  const& get_fill()  const { return std::get<fill_t>(op);  }

    set<int> deps;
    set<int> outs;
  };

  void _verify_deps(set<int> const& deps);

  void _mark_write(int gid, tensor_t& src, tensor_t& dst);

  // Insert a move node
  int move(int src_tensor_id, int dst_tensor_id, set<int> direct_deps = {});

  // Insert a touch node
  int touch(
    touch_t const& op,
    int inn_tensor_id, int out_tensor_id,
    set<int> direct_deps = {});

  // Allocate a temporary tensor
  int alloc(int loc, vector<uint64_t> shape);
  void alloc(int tensor_id, int loc, vector<uint64_t> shape);

  int alloc_(int loc, vector<uint64_t> shape, tensor_type_t type);
  void alloc_(int tensor_id, int loc, vector<uint64_t> shape, tensor_type_t type);

  int new_tensor_id() const;

  int num_locations() const;

  void print_graphviz(std::ostream& out) const;

  int tensor_loc(int tensor_id) const { return tensors.at(tensor_id).loc(); }

  set<int> out_tensors() const;

  int insert_node(
    std::variant<touch_t, move_t, fill_t> op,
    int inn_tensor_id,
    int out_tensor_id,
    set<int> const& deps);

  // Touch the output tensor with the corresponding input tensor.
  // If the tensors are not at the same location, insert alloc tensors,
  // subset ops and move nodes as needed.
  int touch_unto(
    touch_t const& op,
    int inn_tensor_id, int out_tensor_id,
    set<int> direct_deps = {});

  // all tensors an ops are with respect to this dtype
  dtype_t const dtype;
  // all touches, if they have a castable, should be with this
  // castable
  optional<castable_t> const castable;

  vector<node_t> nodes;
  map<int, tensor_t> tensors;
};

std::ostream& operator<<(std::ostream& out, graph_t::tensor_type_t const& tt);
