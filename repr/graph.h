#pragma once
#include "setup.h"

#include "touch.h"

struct graph_t {
  enum tensor_type_t {
    tt_inn,
    tt_out,
    tt_tmp
  };

  struct tensor_t {
    int loc;
    vector<uint64_t> shape;
    tensor_type_t type;

    // all graph ids that have writen to this tensor
    set<int> writes;

    int rank() const { return shape.size(); }
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

  // Insert a move node
  int move(int src_tensor_id, int dst_tensor_id, set<int> direct_deps = {});

  // Insert a touch node
  int touch(
    touch_t const& op,
    int inn_tensor_id, int out_tensor_id,
    set<int> direct_deps = {});

  // Allocate a temporary tensor
  int alloc(int loc, vector<uint64_t> shape);
  int alloc_(int loc, vector<uint64_t> shape, tensor_type_t type);

  // Touch the output tensor with the corresponding input tensor.
  // If the tensors are not at the same location, insert alloc tensors,
  // subset ops and move nodes as needed.
  int touch_unto(
    touch_t const& op,
    int inn_tensor_id, int out_tensor_id,
    set<int> direct_deps = {});

  vector<node_t> nodes;
  vector<tensor_t> tensors;
};
