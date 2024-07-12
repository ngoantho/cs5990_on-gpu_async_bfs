#include "harmonize.git/harmonize/cpp/harmonize.h"
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace util;

typedef struct {
  int id;
  int edge_count;
  int* edge_arr;
  bool visited;
  int depth;
} Node;

// state that will be stored per program instance and accessible by all work
// groups immutable, but can contain references and pointers to non-const data
struct MyDeviceState {
  Node *node_arr;
  bool done;
  int node_count;
};

struct MyProgramOp {
  using Type = void (*)(int level);

  template <typename Program>
  __device__ static void eval(Program program, int level) {
    int this_id = blockIdx.x * blockDim.x + threadIdx.x;
    Node node = program.device.node_arr[this_id];
    printf("depth %d\n", node.depth);
    if (node.depth != level) return;
    else if (node.edge_count == 0) return;

    printf("node %d, depth %d\n", node.id, node.depth);
    program.device.done = false;
    for (int i = 0; i < node.edge_count; i++)
    {
      int edge_id = node.edge_arr[i];
      Node edge_node = program.device.node_arr[edge_id];
      printf("edge %d, depth %d->%d\n", edge_node.id, edge_node.depth, node.depth+1);
      edge_node.depth = node.depth + 1;
    }

    /*
    int this_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (this_id != id) return;
    Node node = program.device.node_arr[id];
    printf("id: %d\n", node.id);
    printf("edge count: %d\n", node.edge_count);
    for (int i = 0; i < node.edge_count; i++)
    {
      printf("next id: %d\n", node.edge_arr[i]);
      program.template async<MyProgramOp>(node.edge_arr[i]);
    }
    */
  }
};

struct MyProgramSpec {
  typedef OpUnion<MyProgramOp> OpSet;
  typedef MyDeviceState DeviceState;

  /*
    type Program {
      device: DeviceState
      template: Op
    }
  */

  // called by each work group at start
  template <typename Program>
  __device__ static void initialize(Program program) {}

  // called by each work group at end
  template <typename Program>
  __device__ static void finalize(Program program) {}

  // called by each work group if need work
  template <typename Program>
  __device__ static bool make_work(Program program) {
    int level = 1;
    do {
      printf("level %d\n", level);
      program.device.done = true;
      program.template async<MyProgramOp>(level);
      level++;
    } while (!program.device.done);
    return program.device.done;
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
  ds.done = false;

  std::vector<Node> nodes = {
      {.id = 0, .edge_count = 3, .edge_arr = nullptr, .visited = false, .depth = -1},
      {.id = 1, .edge_count = 2, .edge_arr = nullptr, .visited = false, .depth = -1},
      {.id = 2, .edge_count = 1, .edge_arr = nullptr, .visited = false, .depth = -1},
      {.id = 3, .edge_count = 1, .edge_arr = nullptr, .visited = false, .depth = -1},
      {.id = 4, .edge_count = 0, .edge_arr = nullptr, .visited = false, .depth = -1},
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
  exec<ProgType>(instance, batch_count, run_count);
  cudaDeviceSynchronize();
  host::check_error();
}