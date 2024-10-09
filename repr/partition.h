#pragma once
#include "../utils/setup.h"
#include "../utils/hrect.h"

#include "partdim.h"
#include "touch.h"

struct partition_t {
  vector<partdim_t> partdims;

  static partition_t singleton(vector<uint64_t> shape);

  static partition_t intersect(partition_t lhs, partition_t rhs);

  vector<uint64_t> total_shape() const;

  int num_parts() const;

  vector<int> block_shape() const;

  int rank() const;

  hrect_t<uint64_t>
  get_region(vector<int> const& idxs) const;
  hrect_t<uint64_t>
  get_region(int block) const;

  hrect_t<int>
  get_exact_covering_blocks(
    hrect_t<uint64_t> const& region) const;

  hrect_t<int>
  get_covering_blocks(
    hrect_t<uint64_t> const& region) const;

  // If multiple index cover the area given by hrect,
  // throw an error
  vector<int> get_covering_block(
    hrect_t<uint64_t> const& region) const;

  tuple<vector<int>, touch_t>
  subset_covering_block(hrect_t<uint64_t> const& region) const;

  vector<int> block_to_index(int block) const;
  int index_to_block(vector<int> const& index) const;
};

tuple<vector<int>, vector<int>, touch_t>
touch_from_covered_region(
  hrect_t<uint64_t> region,
  partition_t const& inn,
  partition_t const& out);

std::function<hrect_t<int>(vector<int> const&)>
build_get_refi_index_region(
  partition_t const& coarse_part,
  partition_t const& refined_part);

bool operator==(partition_t const& lhs, partition_t const& rhs);
bool operator!=(partition_t const& lhs, partition_t const& rhs);

std::ostream& operator<<(std::ostream& out, partition_t const& p);

