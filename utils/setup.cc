#include "setup.h"

std::mt19937& random_gen() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return gen;
}

void set_seed(int seed) {
  random_gen() = std::mt19937(seed);
}

// Stolen from http://myeyesareblind.com/2017/02/06/Combine-hash-values/
// where this is the boost implementation
void hash_combine_impl(std::size_t& seed, std::size_t value)
{
  seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

vector<int> divide_evenly_int(int num_parts, int n) {
  if(num_parts <= 0 || n <= 0) {
    throw std::runtime_error("invalid args");
  }
  if(n < num_parts) {
    throw std::runtime_error("Cannot have size zero parts");
  }
  vector<int> ret(num_parts, n / num_parts);
  int d = n % num_parts;
  for(int i = 0; i != d; ++i) {
    ret[i]++;
  }
  return ret;
}

vector<uint64_t> divide_evenly(int num_parts, uint64_t n) {
  if(n < num_parts) {
    throw std::runtime_error("Cannot have size zero parts");
  }
  vector<uint64_t> ret(num_parts, n / num_parts);
  uint64_t d = n % num_parts;
  for(int i = 0; i != d; ++i) {
    ret[i]++;
  }
  return ret;
}

void* increment_void_ptr(void* ptr, uint64_t size)
{
  return static_cast<void*>(
    static_cast<uint8_t*>(ptr) + size);
}

void const* increment_void_ptr(void const* ptr, uint64_t size)
{
  return static_cast<void const*>(
    static_cast<uint8_t const*>(ptr) + size);
}
