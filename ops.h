#pragma once
#include "setup.h"

#include "transform.h"

// Together, reduce_adjacent and broadcast_adjacent can be
// used to construct a tree-allreduce for each block

// Attempt the following; on failure return None
// For each block:
//   group the partials and send the second member of each group
//   to the first.
//
//   Example: device order is 1 2 3 4 5 6 7 8
//            partials at     x x x x x x x x
//            results in      x   x   x   x
//
//   Example: device order is 1 2 3 4 5 6 7 8
//            partials at     x   x   x   x
//            results in      x       x
//
//   Example: device order is 1 2 3 4 5 6 7
//            partials at     x   x   x x x
//            results in      x   x   x   x
//
// Return None if:
// 1. any partial is at multiple locations
// 2. any pair of partials are at the same location
// 3. any pair of blocks results in a different number of partials
optional<tuple<transform_t, placement_t>>
reduce_adjacent(
  vector<int> device_order,
  placement_t const& inn_pl);

optional<tuple<transform_t, placement_t>>
broadcast_adjacent(
  vector<int> device_order,
  placement_t const& inn_pl);



