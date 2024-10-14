#include "allocator.h"

allocator_t::allocator_t(uint64_t memsize, uint64_t p)
  : alignment_power(p)
{
  blocks.push_back(block_t {
    .beg = 0,
    .end = memsize, 
    .occupied = false
 });
}

// return the smallest value greater than or equal to number
// that is divisible by 2^power.
uint64_t align_to_power_of_two(uint64_t number, uint8_t power)
{
  if(number == 0) {
    return number;
  }

  number--;
  for(int i = 0; i != power; ++i) {
    number |= (1 << i);
  }
  number++;

  return number;
}

uint64_t allocator_t::alloc(uint64_t size) {
  uint64_t ret;
  int which_block = 0;
  for(; which_block != blocks.size(); ++which_block) {
    block_t const& block = blocks[which_block];
    if(!block.occupied) {
      ret = align_to_power_of_two(block.beg, alignment_power);
      if(ret + size <= block.end) {
        break;
      }
    }
  }

  if(which_block == blocks.size()) {
    throw std::runtime_error("could not allocate");
  }

  // Update which block
  uint64_t which_end_time_zero;
  {
    block_t& b = blocks[which_block];
    which_end_time_zero = b.end;
    b.end = ret + size;
    b.occupied = true;
  }

  if(ret + size < which_end_time_zero) {
    // For the remainder, either insert the last block 
    // or update the next block
    if(which_block + 1 == blocks.size()) {
      blocks.push_back(block_t {
        .beg = ret + size,
        .end = which_end_time_zero,
        .occupied = false
      });
    } else {
      block_t& b = blocks[which_block + 1];
      if(b.beg != which_end_time_zero) {
        throw std::runtime_error("invalid block beg found");
      }
      if(b.occupied) {
        throw std::runtime_error("next block is occupied!");
      }
      // Prepend the remainder of which block onto this next block
      b.beg = ret + size;
    }
  }

  return ret;
}

void allocator_t::free(uint64_t offset) {
  auto iter = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk)
    {
      return blk.beg <= offset;
    }
  );

  if(iter == blocks.end())
  {
    throw std::runtime_error("did not find a block");
  }
  if(!iter->occupied) {
    throw std::runtime_error("this block is not occupied!");
  }

  uint64_t s_beg = iter->beg;
  uint64_t s_end = iter->end;
  iter = blocks.erase(iter);

  // erase all subsequent free blocks, updating s_end
  while(iter != blocks.end() && !iter->occupied) {
    s_end = iter->end;
    iter = blocks.erase(iter);
  }

  // Now make sure we have s_beg to s_end
  bool do_insert = true;
  if(iter != blocks.begin()) {
    auto left = iter - 1;
    if(!left->occupied) {
      left->end = s_end;
      do_insert = false;
    }
  } 
  if(do_insert) {
    blocks.insert(iter, block_t {
      .beg = s_beg,
      .end = s_end,
      .occupied = false
    });
  }
};

void allocator_t::print(std::ostream& out) const {
  auto p = [&](block_t const& b) {
    out << "[" << b.beg << "," << b.end << ")." << std::boolalpha << b.occupied;
  };
  p(blocks[0]);
  for(int i = 1; i != blocks.size(); ++i) {
    out << ", ";
    p(blocks[i]);
  }
}

std::ostream& operator<<(std::ostream& out, allocator_t const& a) {
  a.print(out);
  return out;
}

