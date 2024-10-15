#pragma once
#include "setup.h"

struct allocator_t {
  allocator_t() = delete;
  allocator_t(uint64_t memsize, uint64_t alignment_power = 16);

  uint64_t alloc(uint64_t size);

  void free(uint64_t offset);

  void print(std::ostream& out) const;

  void clear(); // free everything
private:
  struct block_t {
    uint64_t beg;
    uint64_t end;
    bool occupied;
  };

  vector<block_t> blocks;
  uint64_t alignment_power;
};

std::ostream& operator<<(std::ostream& out, allocator_t const& a);

