#include "run.h"

#include "touch.h"
#include "fill.h"

#include "wrap_cuda.h"

#ifdef USE_CUDA_PROFILE
#include <cuda_profiler_api.h>
#endif

// Each completed node has a stream and/or an event.
// If you depend on a node that has a stream, you can own that straem.
// Otherwise you must depend on it's event
//
// Note that this is complicated by the fact that for moves, the src gpu gets
// the stream.
//
// As for stream and event creation:
// * all streams and events are created as needed
// * all streams and events are destroyed on destruction
struct run_state_t {
  run_state_t(graph_t const& graph, map<int, mem_t> const& mems)
  : graph(graph), mems(mems)
  {
    int ngpus = graph.num_locations();
    all_streams = vector<vector<cudaStream_t>>(ngpus);
    all_events  = vector<vector<cudaEvent_t>>(ngpus);
  }

  ~run_state_t() {
    for(int device = 0; device != all_streams.size(); ++device) {
      _cuda_set_device(device);
      for(cudaStream_t stream: all_streams[device]) {
        _cuda_destroy_stream(stream);
      }
      for(cudaEvent_t event: all_events[device]) {
        _cuda_destroy_event(event);
      }
    }
  }

  struct run_node_info_t {
    int device;
    cudaStream_t stream;
    vector<cudaEvent_t> events;
    bool needs_event;
  };

  void run() {
    for(int node_id = 0; node_id != graph.nodes.size(); ++node_id) {
      run_node_info_t run_info = get_run_info(node_id);
      auto const& device      = run_info.device;
      auto const& stream      = run_info.stream;
      auto const& events      = run_info.events;
      auto const& needs_event = run_info.needs_event;

      // Note: all the methods/functions in here that take a stream as input,
      //       we assume that cuda is set to the streams device
      _cuda_set_device(device);

      for(cudaEvent_t event: events) {
        _cuda_wait(stream, event);
      }

      execute(stream, node_id);

      optional<cudaEvent_t> fini_event;
      if(needs_event) {
        fini_event = create_and_record_event(device, stream);
      }

      node_infos.push_back(node_info_t {
        .device = device,
        .maybe_stream = stream,
        .maybe_event = fini_event
      });
    }
  }

  run_node_info_t get_run_info(int node_id) {
    int device = get_device(node_id);

    // A node can depend on a another node via
    // either a stream or an event.

    auto const& node = graph.nodes[node_id];
    cudaStream_t stream;
    vector<cudaEvent_t> events;
    {
      bool acquired = false;
      for(int const& inn: node.deps) {
        auto& inn_info = node_infos[inn];

        if(!acquired && bool(inn_info.maybe_stream) && device == get_device(inn)) {
          stream = inn_info.maybe_stream.value();
          inn_info.maybe_stream = std::nullopt;
          acquired = true;
        } else {
          events.push_back(inn_info.maybe_event.value());
        }
      }
      if(!acquired) {
        _cuda_set_device(device);
        stream = create_stream(device);
      }
    }

    // * If there are no outgoing nodes, an event is not needed
    // * If there are multiple nodes that depend on this node, atleast one of them
    //   will need to depend on this node via an event
    // * If there is one outgoing node and it can reuse this stream, we won't need
    //   an event
    bool needs_event;
    if(node.outs.size() == 0) {
      needs_event = false;
    } else if(node.outs.size() == 1) {
      int const& out = *node.outs.begin();
      if(device == get_device(out)) {
        needs_event = false;
      } else {
        needs_event = true;
      }
    } else {
      needs_event = true;
    }

    return run_node_info_t {
      .device = device,
      .stream = stream,
      .events = events,
      .needs_event = needs_event
    };
  }

  int get_device(int node_id) {
    auto const& node = graph.nodes[node_id];
    // Note: graph_t::tensor_loc requires a std::map access which
    //       isn't constant time
    if(node.is_touch()) {
      return graph.tensor_loc(node.out_tensor_id);
    } else if(node.is_move()) {
      return node.get_move().src_loc;
    } else if(node.is_fill()) {
      return graph.tensor_loc(node.out_tensor_id);
    } else {
      throw std::runtime_error("invalid");
    }
  }

  void* get_data(int tensor_id) {
    return mems.at(tensor_id).data;
  }

  void execute(cudaStream_t stream, int node_id) {
    auto const& node = graph.nodes[node_id];
    if(node.is_touch()) {
      execute_touch(
        stream,
        node.get_touch(),
        get_data(node.out_tensor_id),
        get_data(node.inn_tensor_id));
    } else if(node.is_move()) {
      _cuda_move(
        stream,
        dtype_size(graph.dtype) * node.get_move().elem,
        get_data(node.out_tensor_id),
        get_data(node.inn_tensor_id));
    } else if(node.is_fill()) {
      auto const& fill = node.get_fill();
      execute_fill(
        stream,
        fill.scalar,
        fill.elem,
        get_data(node.out_tensor_id));
    } else {
      throw std::runtime_error("invalid");
    }
  }

  cudaEvent_t create_and_record_event(int device, cudaStream_t stream) {
    auto& ee = all_events[device];
    ee.emplace_back();
    cudaEvent_t& event = ee.back();
    _cuda_handle_error(
      cudaEventCreate(&event),
      "create event");

    _cuda_handle_error(
      cudaEventRecord(event, stream),
      "event record");

    return event;
  }

  cudaStream_t create_stream(int device) {
    auto& ss = all_streams[device];
    ss.emplace_back();
    cudaStream_t& ret = ss.back();
    _cuda_handle_error(
      cudaStreamCreate(&ret),
      "create stream");
    return ret;
  }

  graph_t const& graph;
  map<int, mem_t> const& mems;

  struct node_info_t {
    int device;
    optional<cudaStream_t> maybe_stream;
    optional<cudaEvent_t> maybe_event;
  };

  vector<node_info_t> node_infos;

  vector<vector<cudaStream_t>> all_streams;
  vector<vector<cudaEvent_t>> all_events;
};

void run_graph(
  graph_t const& graph,
  map<int, mem_t> const& mems)
{
#ifdef USE_CUDA_PROFILE
  _cuda_handle_error(cudaProfilerStart());
#endif

  // Make sure each tensor in the graph is in mems, at the right place,
  // with the enough size
  for(auto const& [tid, tensor]: graph.tensors) {
    auto iter = mems.find(tid);
    if(iter == mems.end()) {
      throw std::runtime_error("tid " + write_with_ss(tid) + " not in mems");
    }
    mem_t const& mem = iter->second;

    if(mem.device != tensor.loc()) {
      throw std::runtime_error("tensor loc != mem device");
    }
    if(mem.size < product(tensor.shape())*sizeof(float)) {
      throw std::runtime_error("memory size not large enough");
    }
  }

  run_state_t state(graph, mems);
  state.run();

#ifdef USE_CUDA_PROFILE
  _cuda_handle_error(cudaProfilerStop());
#endif
}
