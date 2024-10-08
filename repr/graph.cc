#include "graph.h"

graph_t::tensor_t::tensor_t()
  : tensor_t(-1, {}, tt_tmp)
{}

graph_t::tensor_t::tensor_t(int l, vector<uint64_t> s, graph_t::tensor_type_t t)
  : _loc(l), _shape(s), _type(t), read_only(false)
{
  if(_type == tt_inn) {
    read_only = true;
  }
}

void graph_t::tensor_t::insert_write(int gid) {
  if(read_only) {
    throw std::runtime_error("cannot write to read only tensor");
  }
  _writes.insert(gid);
}

void graph_t::tensor_t::write_with() {
  // Once you write with a tensor, it is always read only
  read_only = true;
}

void graph_t::_verify_deps(set<int> const& deps) {
  for(int const& inn: deps) {
    if(inn < 0 || inn >= nodes.size()) {
      throw std::runtime_error("invalid dep..");
    }
  }
}

void graph_t::_mark_write(
  int gid,
  graph_t::tensor_t& src,
  graph_t::tensor_t& dst)
{
  src.write_with();
  dst.insert_write(gid);
}

int graph_t::move(int src_tensor_id, int dst_tensor_id, set<int> deps) {
  tensor_t & src = tensors.at(src_tensor_id);
  tensor_t & dst = tensors.at(dst_tensor_id);

  set_append(deps, src.writes());

  if(dst.is_read_only()) {
    throw std::runtime_error("move: cannot write to read-only tensors");
  }

  if(!vector_equal(src.shape(), dst.shape())) {
    throw std::runtime_error("cannot move tensors of different shape");
  }

  if(src.loc() == dst.loc()) {
    throw std::runtime_error("why move tensor to the same location?");
  }

  _verify_deps(deps);

  move_t move {
    .src_loc = src.loc(),
    .dst_loc = dst.loc(),
    .size = product(src.shape())
  };

  nodes.push_back(node_t {
    .op = move,
    .inn_tensor_id = src_tensor_id,
    .out_tensor_id = dst_tensor_id,
    .deps = deps
  });

  int ret = nodes.size() - 1;

  _mark_write(ret, src, dst);

  return ret;
}

int graph_t::touch(
  touch_t const& op,
  int inn_tensor_id, int out_tensor_id,
  set<int> deps)
{
  if(inn_tensor_id == out_tensor_id) {
    throw std::runtime_error("out cannot be in for touch op");
  }

  tensor_t & inn = tensors.at(inn_tensor_id);
  tensor_t & out = tensors.at(out_tensor_id);

  set_append(deps, inn.writes());

  if(out.is_read_only()) {
    throw std::runtime_error("touch: cannot write to read-only tensors");
  }

  if(inn.loc() != out.loc()) {
    throw std::runtime_error("all touch ops must be local!");
  }

  if(!vector_equal(inn.shape(), op.inn_shape())) {
    throw std::runtime_error("invlaid inn sahpe for this touch");
  }

  if(!vector_equal(out.shape(), op.out_shape())) {
    throw std::runtime_error("invlaid out sahpe for this touch");
  }

  _verify_deps(deps);

  nodes.push_back(node_t {
    .op = op,
    .inn_tensor_id = inn_tensor_id,
    .out_tensor_id = out_tensor_id,
    .deps = deps
  });

  int ret = nodes.size() - 1;

  _mark_write(ret, inn, out);

  return ret;
}

int graph_t::alloc(int loc, vector<uint64_t> shape) {
  return alloc_(loc, shape, tt_tmp);
}

int graph_t::alloc_(int loc, vector<uint64_t> shape, tensor_type_t tt) {
  tensors.emplace_back(loc, shape, tt);
  return tensors.size() - 1;
}

int graph_t::touch_unto(
  touch_t const& op,
  int inn_tensor_id, int out_tensor_id,
  set<int> curr_deps)
{
  tensor_t const& inn = tensors.at(inn_tensor_id);
  tensor_t const& out = tensors.at(out_tensor_id);

  if(inn.loc() == out.loc()) {
    // this is just a touch!
    return touch(op, inn_tensor_id, out_tensor_id, curr_deps);
  }

  set_append(curr_deps, inn.writes());
  _verify_deps(curr_deps);

  // Since the tensors are at different locations, we must
  // move some data...
  // In the worst case case:
  // 1. subset the input tensor
  // 2. move that subset to destination subset
  // 3. touch the destination subset into the output

  int move_src_tensor = inn_tensor_id;
  if(!op.uses_full_inn()) {
    // allocate temporary memory for moving from and
    // write into it
    move_src_tensor = alloc(inn.loc(), op.write_shape());
    int x = touch(
      op.inn_to_write(),
      inn_tensor_id, move_src_tensor,
      curr_deps);
    curr_deps = set<int>({x});
  }

  int move_dst_tensor = out_tensor_id;
  if(!op.uses_full_out()) {
    // allocate temporary memory for moving into
    move_dst_tensor = alloc(out.loc(), op.write_shape());
  }

  // do the move
  {
    int x = move(move_src_tensor, move_dst_tensor, curr_deps);
    curr_deps = set<int>{x};
  }

  if(!op.uses_full_out()) {
    // touch into the output (if needed)
    int x = touch(
      op.write_to_out(),
      move_dst_tensor, out_tensor_id,
      curr_deps);
    curr_deps = set<int>{x};
  }

  int ret = *curr_deps.begin();
  return ret;
}
