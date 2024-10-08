#pragma once
#include "../utils/setup.h"
#include "../utils/vtensor.h"

#include "partition.h"
#include "placement.h"

struct relation_t {
  partition_t partition;
  vtensor_t<map<int, int>> locations;

  int num_partials() const;

  bool has_partials() const { return num_partials() > 1; }

  placement_t as_placement() const;
};
