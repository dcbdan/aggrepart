#include "wrap_cuda.h"

void __cuda_handle_error(cudaError_t error, string line, string msg) {
  if(error != cudaSuccess){ 
    DOUT("error is " << int(error) << " ... msg: " << msg);
    if(msg == "") {
      msg = "handle_cuda_error";
    } 
    throw std::runtime_error(line + "\n" + msg + ": " + string(cudaGetErrorString(error)));
  }
}

void _cuda_set_device(int device) {
  _cuda_handle_error(
    cudaSetDevice(device),
    "_cuda_set_device");
}

// Note: cuda streams and events are lightweight, don't bother
//       with const, const& and &

void _cuda_destroy_stream(cudaStream_t stream) {
  _cuda_handle_error(
    cudaStreamDestroy(stream),
    "_cuda_destroy_stream");
}

void _cuda_destroy_event(cudaEvent_t event) {
  _cuda_handle_error(
    cudaEventDestroy(event),
    "_cuda_destroy_error");
}

void _cuda_wait(cudaStream_t stream, cudaEvent_t event) {
  _cuda_handle_error(
    cudaStreamWaitEvent(stream, event),
    "_cuda_wait");
}

void _cuda_move(cudaStream_t stream, uint64_t size, void* dst, void const* src) {
  _cuda_handle_error(
    cudaMemcpyAsync(
      dst, src, size, cudaMemcpyDeviceToDevice, stream),
    "_cuda_move");
}

void* _cuda_malloc(uint64_t size) {
  void* ret;
  _cuda_handle_error(
     cudaMalloc(&ret, size),
     "_cuda_malloc");
  return ret;
}

void _cuda_free(void* data) {
  _cuda_handle_error(
     cudaFree(data),
     "_cuda_free");
}

void _cuda_device_sync() {
  _cuda_handle_error(
      cudaDeviceSynchronize(),
      "_cuda_device_sync");
}

void _cuda_sync_all(int num_devices) {
  for(int i = 0; i != num_devices; ++i) {
    _cuda_set_device(i);
    _cuda_device_sync();
  }
}

int _cuda_enable_peer_access()
{
  int device_count;
  _cuda_handle_error(cudaGetDeviceCount(&device_count));
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i != j) {
        _cuda_set_device(i);
        // enable p2p access
        _cuda_handle_error(cudaDeviceEnablePeerAccess(j, 0));
        // enable host memory mapping access by cudaHostAlloc
        _cuda_handle_error(cudaSetDeviceFlags(cudaDeviceMapHost));
      }
    }
  }

  // check if the peer access is really enabled
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i != j) {
        int canAccessPeer;
        _cuda_set_device(i);
        _cuda_handle_error(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
        if (canAccessPeer != 1){
          throw std::runtime_error("Peer access is not enabled");
        }
      }
    }
  }

	return device_count;
}



