#include "harmonize.git/harmonize/cpp/harmonize.h"
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace util;

typedef struct {
  int id;
  int edge_count; // edges that branch off
  int* edge_arr; // idx of nodes
  int visited; // base case: node is visited
  int depth; // compare depth of this node and incoming depth
} Node;

// state that will be stored per program instance and accessible by all work
// groups immutable, but can contain references and pointers to non-const data
struct MyDeviceState {
  Node *node_arr;
  int node_count;
  // TODO: int pointer
  iter::AtomicIter<unsigned int>* iterator;
};

struct MyProgramOp {
  using Type = void (*)(Node* node);

  template <typename PROGRAM>
  __device__ static void eval(PROGRAM prog, Node* node) {
    // printf("node %d\n", node->id);

    // if this node is already visited, then skip it
    if (atomicAnd(&node->visited, 1)) {
      printf("node %d visited already\n", node->id);
      return; // base case
    }

    // set visited bit to 1    
    atomicAdd(&node->visited, 1);

    // verify visited bit
    if (atomicAnd(&node->visited, 1)) {
      printf("node %d visited=1\n", node->id);
    }

    for (int i = 0; i < node->edge_count; i++) {
      int edge_id = node->edge_arr[i];
      Node& edge_node = prog.device.node_arr[edge_id];
      printf("edge id %d, node id %d\n", edge_id, edge_node.id);
      prog.template async<MyProgramOp>(&edge_node);
    }
  }
};

struct MyProgramSpec {
  typedef OpUnion<MyProgramOp> OpSet;
  typedef MyDeviceState DeviceState;

  static const size_t STASH_SIZE =   16;
	static const size_t FRAME_SIZE = 8191;
	static const size_t  POOL_SIZE = 8191;

  /*
    type PROGRAM {
      device: DeviceState
      template: Op
    }
  */

  // called by each work group at start
  template <typename PROGRAM>
  __device__ static void initialize(PROGRAM prog) {}

  // called by each work group at end
  template <typename PROGRAM>
  __device__ static void finalize(PROGRAM prog) {}

  // called by each work group if need work
  template <typename PROGRAM>
  __device__ static bool make_work(PROGRAM prog) {
    unsigned int index;
    if (prog.device.iterator->step(index)) {
      prog.template async<MyProgramOp>(&prog.device.node_arr[0]);
    }
    return false;
  }
};

using ProgType = AsyncProgram<MyProgramSpec>;

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);

  // arguments
  unsigned int batch_count = args["batch_count"] | 1;
  std::cout << "group count: " << batch_count << std::endl;
  unsigned int run_count = args["run_count"] | 1;
  std::cout << "cycle count: " << run_count << std::endl;
  unsigned int arena_size = args["arena_size"] | 0x10000;
  std::cout << "arena size: " << arena_size << std::endl;

  // init DeviceState
  MyDeviceState ds;
  ds.node_count = 5;

  iter::AtomicIter<unsigned int> host_iter(0, 1);
	host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
	iterator << host_iter;
	ds.iterator = iterator;

  std::vector<Node> nodes = {
      {.id = 0, .edge_count = 3, .edge_arr = nullptr, .visited = 0, .depth = 0},
      {.id = 1, .edge_count = 2, .edge_arr = nullptr, .visited = 0, .depth = 0},
      {.id = 2, .edge_count = 1, .edge_arr = nullptr, .visited = 0, .depth = 0},
      {.id = 3, .edge_count = 1, .edge_arr = nullptr, .visited = 0, .depth = 0},
      {.id = 4, .edge_count = 0, .edge_arr = nullptr, .visited = 0, .depth = 0},
  };
  nodes[0].depth = 1;
  
  std::vector<int> v0 = {2, 3, 4};
  host::DevBuf<int> dev_v0(3);
  dev_v0 << v0;
  nodes[0].edge_arr = dev_v0;

  std::vector<int> v1 = {3, 4};
  host::DevBuf<int> dev_v1(2);
  dev_v1 << v1;
  nodes[1].edge_arr = dev_v1;

  std::vector<int> v2 = {3};
  host::DevBuf<int> dev_v2(1);
  dev_v2 << v2;
  nodes[2].edge_arr = dev_v2;

  std::vector<int> v3 = {4};
  host::DevBuf<int> dev_v3(1);
  dev_v3 << v3;
  nodes[3].edge_arr = dev_v3;

  host::DevBuf<Node> dev_nodes = host::DevBuf<Node>(ds.node_count);
  dev_nodes << nodes;
  ds.node_arr = dev_nodes;
  
  // declare program instance
  ProgType::Instance instance(arena_size, ds);
  cudaDeviceSynchronize();
  host::check_error();

  // init program instance
  init<ProgType>(instance, 32);
  cudaDeviceSynchronize();
  host::check_error();

  // exec program instance
  do {
			// Give the number of work groups and the size of the chunks pulled from
			// the io buffer
			exec<ProgType>(instance,batch_count,run_count);
			cudaDeviceSynchronize();
			host::check_error();
		} while ( ! instance.complete() );
}