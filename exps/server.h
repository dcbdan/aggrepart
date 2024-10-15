#pragma once
#include "../utils/setup.h"
#include "../utils/allocator.h"

#include "../engine/wrap_cuda.h"
#include "../engine/fill.h"
#include "../engine/run.h"

struct server_t {
  server_t() = delete;

  server_t(uint64_t mem_per_device, int num_devices)
    : server_t(vector<uint64_t>(num_devices, mem_per_device))
  {}

  server_t(vector<uint64_t> memsizes) {
    _cuda_enable_peer_access(memsizes.size());

    for(int d = 0; d != memsizes.size(); ++d) {
      uint64_t const& memsize = memsizes[d];
      allocators.emplace_back(memsize);
    
      _cuda_set_device(d);
      ptrs.push_back(_cuda_malloc(memsize));
    }
  }

  ~server_t() {
    for(int d = 0; d != ptrs.size(); ++d) {
      _cuda_set_device(d);
      _cuda_free(ptrs[d]);
    }
  }

  void free(int tensor_id) {
    auto iter = data.find(tensor_id);
    if(iter == data.end()) {
      throw std::runtime_error("not available!");
    }

    mem_t const& m = iter->second;
    uint64_t offset = uint64_t(m.data) - uint64_t(ptrs[m.device]);
    allocators[m.device].free(offset);

    data.erase(iter);
  }

  void allocate(int tensor_id, int device, uint64_t size) {
    uint64_t offset = allocators[device].alloc(size);
    data.insert({ tensor_id, mem_t {
      .data   = increment_void_ptr(ptrs[device], offset),
      .size   = size,
      .device = device
    }});
  }

  void fill(int tensor_id, scalar_t val) {
    mem_t mem = data.at(tensor_id);
    if(mem.size % dtype_size(val.dtype) != 0) {
      throw std::runtime_error("invalid dtype: server fill");
    }

    uint64_t nelem = mem.size / dtype_size(val.dtype);

    _cuda_set_device(mem.device);
    execute_fill(0, val, nelem, mem.data);
  }

  uint64_t nlocs() const {
    return ptrs.size();
  }

  void clear() {
    for(allocator_t& a: allocators) {
      a.clear();
    }
    data = map<int, mem_t>();
  }

  map<int, mem_t> data;
  vector<allocator_t> allocators;
  vector<void*> ptrs;
};


