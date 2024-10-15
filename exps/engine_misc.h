void init_data_to_ones(server_t& server, graph_t const& graph) {
  for(auto const& [tid, tensor]: graph.tensors) {
    int device = tensor.loc();
    uint64_t size = product(tensor.shape())*dtype_size(graph.dtype);
    server.allocate(tid, device, size);

    if(tensor.type() == graph_t::tt_inn) {
      server.fill(tid, scalar_t::make_one(graph.dtype));
    }
  }
}

map<int, tuple<float, float>> 
tensor_ranges(graph_t const& graph, map<int, mem_t> const& gpu_data) 
{
  vector<float> cpu_data;
  map<int, tuple<float, float>> ret;

  for(auto const& [tid, tensor]: graph.tensors) {
    if(tensor.type() != graph_t::tensor_type_t::tt_out) {
      continue;
    }

    mem_t const& mem = gpu_data.at(tid);
    uint64_t nelem = product(tensor.shape());

    cpu_data.resize(nelem);
    _cuda_set_device(mem.device);
    _cuda_handle_error(cudaMemcpy(
      cpu_data.data(), mem.data, mem.size, 
      cudaMemcpyDeviceToHost));

    float mn = cpu_data[0];
    float mx = cpu_data[0];
    for(uint64_t i = 1; i != nelem; ++i) {
      mn = std::min(mn, cpu_data[i]);
      mx = std::max(mx, cpu_data[i]);
    }

    //DOUT(tid << ": " << vector<float>(cpu_data.begin(), cpu_data.begin() + nelem));

    ret.insert({tid, {mn,mx}});
  }

  return ret;
}

