#include "harmonize.git/harmonize/cpp/harmonize.h"
using namespace util;

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "node.h"
#include "adjacency_graph.h"
#include "node_graph.h"

// state that will be stored per program instance and accessible by all work
// groups immutable, but can contain references and pointers to non-const data
struct MyDeviceState {
  Node* node_arr;
  int node_count;
  int* edge_arr;
  int root_node;
  iter::AtomicIter<unsigned int>* iterator;
};

struct MyProgramOp {
  using Type = void (*)(Node* node, unsigned int current_depth, Node* parent);

  template <typename PROGRAM>
  __device__ static void eval(PROGRAM prog, Node *node, unsigned int current_depth, Node* parent) {
    // int this_id = blockIdx.x * blockDim.x + threadIdx.x;

    // if this node is already visited, then skip it

    // use as baseline for CPU version
    atomicCAS(&node->visited, 0, 1);


    // Simpler version, without loop coalescing
    /*
    for (int i = 0; i < node->edge_count; i++) {

      unsigned int edge_node_id = prog.device.edge_arr[node->edge_offset + i];
      Node& edge_node = prog.device.node_arr[edge_node_id];

      if (atomicMin(&edge_node.depth, current_depth+1) > current_depth+1) {
        prog.template async<MyProgramOp>(&edge_node, current_depth + 1, node);
      }
    }
    */

    //*
    for (int i = 0; i < node->edge_count; i++) {
      int edge_node_id;
      bool hit = false;
      while ( (!hit) && (i < node->edge_count) ){
        edge_node_id = prog.device.edge_arr[node->edge_offset + i];
        Node& edge_node = prog.device.node_arr[edge_node_id];
        if (atomicMin(&edge_node.depth, current_depth+1) > current_depth+1) {
          hit = true;
          break;
        }
        i++;
      }
      if ( hit ) {
        Node& edge_node = prog.device.node_arr[edge_node_id];
        prog.template async<MyProgramOp>(&edge_node, current_depth + 1, node);
      }
    }
    //*/

  }


};

struct MyProgramSpec {
  typedef OpUnion<MyProgramOp> OpSet;
  typedef MyDeviceState DeviceState;

  static const size_t STASH_SIZE =   16;
  static const size_t FRAME_SIZE = 8191;
  static const size_t POOL_SIZE  = 8191;

  /*
    type PROGRAM {
      device: DeviceState
      template: Op
    }
  */

  // called by each work group at start
  template <typename PROGRAM> __device__ static void initialize(PROGRAM prog) {}

  // called by each work group at end
  template <typename PROGRAM> __device__ static void finalize(PROGRAM prog) {}

  // called by each work group if need work
  template <typename PROGRAM> __device__ static bool make_work(PROGRAM prog) {
    unsigned int index;

    if (prog.device.iterator->step(index)) {
      Node &root = prog.device.node_arr[prog.device.root_node];
      atomicMin(&root.depth,0);
      prog.template async<MyProgramOp>(&root, 0, nullptr);
    }

    return false;
  }
};

using AsyncProgType = AsyncProgram<MyProgramSpec>;
using EventProgType = EventProgram<MyProgramSpec>;

template<typename ProgType, typename ProgTypeInstance>
void run_kernel(MyDeviceState ds, unsigned int arena_size, unsigned int group_count, unsigned int cycle_count) {
  ProgTypeInstance instance(arena_size, ds);
  cudaDeviceSynchronize();
  host::check_error();

  // init program instance
  init<ProgType>(instance, 32);
  cudaDeviceSynchronize();
  host::check_error();

  // exec program instance
  do {
    // Give the number of work groups and the size of the chunks pulled from the io buffer
    exec<ProgType>(instance, group_count, cycle_count);
    cudaDeviceSynchronize();
    host::check_error();
  } while (!instance.complete());
}

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);

  // arguments
  unsigned int group_count = args["group_count"]; // batch count
  unsigned int cycle_count = args["cycle_count"]; // run count
  unsigned int arena_size = args["arena_size"] | 0x100000; // amount of memory to allocate

  char* file_str = args.get_flag_str((char*)"file");
  if (file_str == nullptr) {
    std::cerr << "no value provided for -file" << std::endl;
    std::exit(1);
  }

  char* program_type = args.get_flag_str((char*)"program");
  if (program_type == nullptr) {
    std::cerr << "no value provided for -program" << std::endl;
    std::exit(1);
  }

  // if flag is present, then true, else false
  bool directed = args["directed"];

  // init DeviceState
  MyDeviceState ds;
  ds.root_node = args["root"]; // int

  iter::AtomicIter<unsigned int> host_iter(0, 1);
  host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
  iterator << host_iter;
  ds.iterator = iterator;

  AdjacencyGraph adjacency_graph(file_str, ds.node_count, directed);
  NodeGraph node_graph(adjacency_graph.data, ds.node_count);
  if (ds.node_count == 0) {
    std::cerr << "error: node count = 0" << std::endl;
    return 0;
  }

  host::DevBuf<int> dev_edges(node_graph.edges.size());
  dev_edges << node_graph.edges;
  ds.edge_arr = dev_edges;

  host::DevBuf<Node> dev_nodes(ds.node_count);
  dev_nodes << node_graph.nodes;
  ds.node_arr = dev_nodes;

  Stopwatch watch;
  watch.start();

  if (std::string(program_type) == "async") {
    run_kernel<AsyncProgType, AsyncProgType::Instance>(ds, arena_size, group_count, cycle_count);
  } else if (std::string(program_type) == "event") {
    run_kernel<EventProgType, EventProgType::Instance>(ds, arena_size, group_count, cycle_count);
  }

  watch.stop();
  float msec = watch.ms_duration();

  bool raw = args["raw"];
  if (raw) std::cout << msec << std::endl;
  else std::cout << "Runtime: " << msec << "ms" << std::endl;

  std::vector<Node> output;
  dev_nodes >> output;
  for (Node& node : output) {
    std::cout << node.id << "\t" << node.depth << std::endl;
  }

}