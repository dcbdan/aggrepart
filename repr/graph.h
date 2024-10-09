#pragma once
#include "../utils/setup.h"

#include "touch.h"

struct graph_t {
  enum tensor_type_t {
    tt_inn,
    tt_out,
    tt_tmp
  };

  struct tensor_t {
    tensor_t();

    tensor_t(int loc, vector<uint64_t> shape, tensor_type_t type);

    int loc() const { return _loc; }

    vector<uint64_t> const& shape() { return _shape; }
    int rank() const { return _shape.size(); }

    tensor_type_t type() const { return _type; }

    void write_with();

    void insert_write(int gid);

    bool is_read_only() const { return read_only; }

    set<int> writes() const { return _writes; }
  private:
    int _loc;
    vector<uint64_t> _shape;
    tensor_type_t _type;
    bool read_only;

    // all graph ids that have writen to this tensor
    set<int> _writes;
  };

  struct move_t {
    int src_loc;
    int dst_loc;
    uint64_t size;
  };

  struct node_t {
    std::variant<touch_t, move_t> op;
    int inn_tensor_id;
    int out_tensor_id;

    set<int> deps;
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

  // Touch the output tensor with the corresponding input tensor.
  // If the tensors are not at the same location, insert alloc tensors,
  // subset ops and move nodes as needed.
  int touch_unto(
    touch_t const& op,
    int inn_tensor_id, int out_tensor_id,
    set<int> direct_deps = {});

  vector<node_t> nodes;
  map<int, tensor_t> tensors;
};
