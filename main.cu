#include <iostream>
#include <stdio.h>
#include "harmonize.git/harmonize/cpp/harmonize.h"
using namespace util;

typedef struct {
  int start; // index of first adjacent node
  int length; // number of adjacent nodes
  int distance;
} Node;

// state that will be stored per program instance and accessible by all work groups
// immutable, but can contain references and pointers to non-const data
struct MyDeviceState
{
  Node* nodes;
  int* edges;
  int* next_up;
  int* visited;
  int* cost; // TODO: remove
  int node_count;
};

struct MyProgram {
  using Type = void(*)();

  template<typename Program>
  __device__ static void eval(Program program) { // TODO: recusion
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("node %d\n", id);

    if (program.device.next_up[id] == 1 && program.device.visited[id] == 0) {
      program.device.next_up[id] = 0;
      program.device.visited[id] = 1;
      // __syncthreads();
      int start = program.device.nodes[id].start;
      int end = start + program.device.nodes[id].length;
      printf("start: %d, end: %d\n", start, end);
      for (int i = start; i < end; i++) {
        int nid = program.device.edges[i];
        printf("adjacent %d\n", nid);
        if (program.device.visited[nid] == 0) {
          printf("not visited %d\n", nid);
          program.device.cost[nid] = program.device.cost[id] + 1;
          program.device.next_up[nid] = 1;
        }
      }
    }
  }
};

struct MyProgramSpec {
  typedef OpUnion<MyProgram> OpSet;
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
    program.template async<MyProgram>();
    return true;
  }
};

using ProgType = AsyncProgram<MyProgramSpec>;

int main(int argc, char *argv[])
{
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
  ds.node_count = 32;

  std::vector<Node> nodes = {
      {.start = 0, .length = 2}, // 0
      {.start = 2, .length = 1}, // 1
      {.start = 3, .length = 1},
      {.start = 4, .length = 1},
      {.start = 5, .length = 0},
      {.start = 6, .length = 0},
      {.start = 7, .length = 0},
      {.start = 8, .length = 0},
      {.start = 9, .length = 0},
      {.start = 10, .length = 0},
      {.start = 11, .length = 0},
      {.start = 12, .length = 0},
      {.start = 13, .length = 0},
      {.start = 14, .length = 0},
      {.start = 15, .length = 0},
      {.start = 16, .length = 0},
      {.start = 17, .length = 0},
      {.start = 18, .length = 0},
      {.start = 19, .length = 0},
      {.start = 20, .length = 0},
      {.start = 21, .length = 0},
      {.start = 22, .length = 0},
      {.start = 23, .length = 0},
      {.start = 24, .length = 0},
      {.start = 25, .length = 0},
      {.start = 26, .length = 0},
      {.start = 27, .length = 0},
      {.start = 28, .length = 0},
      {.start = 29, .length = 0},
      {.start = 30, .length = 0},
      {.start = 31, .length = 0},
      {.start = 32, .length = 0},
  };
  host::DevBuf<Node> dev_nodes = host::DevBuf<Node>(ds.node_count);
  dev_nodes << nodes;
  ds.nodes = dev_nodes;

  std::vector<int> edges(ds.node_count);
  edges.at(0) = 1; // 0
  edges.at(1) = 4;
  edges.at(2) = 3;
  edges.at(3) = 4;
  edges.at(4) = 4;
  edges.at(5) = 5;
  edges.at(6) = 6;
  edges.at(7) = 7;
  edges.at(8) = 8;
  edges.at(9) = 9;
  edges.at(10) = 10;
  edges.at(11) = 11;
  edges.at(12) = 12;
  edges.at(13) = 13;
  edges.at(14) = 14;
  edges.at(15) = 15;
  edges.at(16) = 16;
  edges.at(17) = 17;
  edges.at(18) = 18;
  edges.at(19) = 19;
  edges.at(20) = 20;
  edges.at(21) = 21;
  edges.at(22) = 22;
  edges.at(23) = 23;
  edges.at(24) = 24;
  edges.at(25) = 25;
  edges.at(26) = 26;
  edges.at(27) = 27;
  edges.at(28) = 28;
  edges.at(29) = 29;
  edges.at(30) = 30;
  edges.at(31) = 31;
  host::DevBuf<int> dev_edges = host::DevBuf<int>(ds.node_count);
  dev_edges << edges;
  ds.edges = dev_edges;

  std::vector<int> next_up(ds.node_count, 0);
  int source = 0;
  next_up[source] = 1;
  host::DevBuf<int> dev_next_up = host::DevBuf<int>(ds.node_count);
  dev_next_up << next_up;
  ds.next_up = dev_next_up;

  std::vector<int> visited(ds.node_count, 0);
  host::DevBuf<int> dev_visited = host::DevBuf<int>(ds.node_count);
  dev_visited << visited;
  ds.visited = dev_visited;

  std::vector<int> cost(ds.node_count, 0);
  host::DevBuf<int> dev_cost = host::DevBuf<int>(ds.node_count);
  dev_cost << cost;
  ds.cost = dev_cost;

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

  std::cout << "cost: node = ?" << std::endl;
  std::vector<int> host_cost(ds.node_count);
  dev_cost >> host_cost;
  for (size_t i = 0; i < ds.node_count; i++)
  {
    std::cout << i << " = " << host_cost[i] << std::endl;
  }
}