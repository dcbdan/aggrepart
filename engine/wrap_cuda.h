#pragma once
#include "../utils/setup.h"

#include <cuda_runtime.h>

#define _cuda_handle_error(error, ...) \
  __cuda_handle_error( \
    error, \
    write_with_ss(__FILE__) + ":" + write_with_ss(__LINE__), \
    ##__VA_ARGS__); 

void __cuda_handle_error(cudaError_t error, string line, string msg = "");

void _cuda_set_device(int device);

// Note: cuda streams and events are lightweight, don't bother
//       with const, const& and &

void _cuda_destroy_stream(cudaStream_t stream);

void _cuda_destroy_event(cudaEvent_t event);

void _cuda_wait(cudaStream_t stream, cudaEvent_t event);

void _cuda_move(cudaStream_t stream, uint64_t size, void* dst, void const* src);

void* _cuda_malloc(uint64_t size);

void _cuda_free(void* data);

void _cuda_device_sync();

void _cuda_sync_all(int num_devices);

// return the number of devices peer access enabled for
int _cuda_enable_peer_access(int num_devices = 0);
