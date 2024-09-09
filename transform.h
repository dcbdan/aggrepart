#pragma once
#include "setup.h"

#include "placement.h"

struct transform_t {
  struct piece_t {
    int block;
    int partial;
    int loc;
  };
  struct convert_t {
    piece_t inn;
    piece_t out;
  };
  vector<convert_t> ops;

  static bool valid(
    placement_t const& inn_pl,
    placement_t const& out_pl,
    transform_t const& ops);

  static transform_t make_naive_transform(
    placement_t const& inn_pl,
    placement_t const& out_pl);

  struct move_t {
    int src;
    int dst;
    uint64_t size;
  };
  static vector<move_t> make_moves(
    placement_t const& inn_pl,
    placement_t const& out_pl,
    transform_t const& ops);
};


