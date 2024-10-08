#pragma once
#include "setup.h"

#include "vtensor.h"
#include "partition.h"
#include "placement.h"

struct relation_t {
  partition_t partition;
  vtensor_t<map<int, int>> locations;

  int num_duplicates() const; // TODO

  bool has_duplicates() const { return num_duplicates > 1; }

  placement_t as_placement() const;
};
