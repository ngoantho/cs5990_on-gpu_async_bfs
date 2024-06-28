#include <iostream>
#include <stdio.h>
#include "graph.h"
#include "harmonize.git/harmonize/cpp/harmonize.h"
using namespace util;

// state that will be stored per program instance and accessible by all work groups
// immutable, but can contain references and pointers to non-const data
struct MyDeviceState
{
  int* adjacencyList;
  int* edgesOffset;
  int* edgesSize;
  int* distance;
  int* parent;
  int N;
};

struct BFSProgram {
  using Type = void(*)(int);

  template<typename Program>
  __device__ static void eval(Program program, int level) {
    int this_id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d:distance=%d\n", this_id, program.device.distance[this_id]);
    // printf("%d:edgesOffset=%d\n", this_id, program.device.edgesOffset[this_id]);
    // printf("%d:edgesSize=%d\n", this_id, program.device.edgesSize[this_id]);
    if (this_id < program.device.N && program.device.distance[this_id] == level) {
      for (int i = program.device.edgesOffset[this_id]; i < program.device.edgesOffset[this_id] + program.device.edgesSize[this_id]; i++) {
        int edge = program.device.adjacencyList[i];
        if (level + 1 < program.device.distance[edge]) {
          program.device.distance[edge] = level + 1;
          program.device.parent[edge] = i;
          printf("%d: distance[%d] = %d\n", this_id, edge, level+1);
          printf("%d: parent[%d] = %d\n", this_id, edge, i);
        }
      }
    }
  }
};

struct MyProgramSpec {
  typedef OpUnion<BFSProgram> OpSet;
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
    int level = 0;
    program.template async<BFSProgram>(level);
    return true;
    /*
    unsigned int iter_step_length = 1u;
    iter::Iter<unsigned int> iter = program.device.iterator->leap(iter_step_length);

    unsigned int index = 0;
    while (iter.step(index)) {
      program.template async<BFSProgram>(index);
    }
    return !program.device.iterator->done();
    */
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

  int numVertices = args["num_vertices"];
  int numEdges = args["num_edges"];
  Graph G(numVertices, numEdges);
  
  unsigned int startVertex = args["start_vertex"];
  std::cout << "start vertex: " << startVertex << std::endl;

  // init DeviceState
  MyDeviceState ds;
  ds.N = numVertices;

  std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
  distance[startVertex] = 0;
  host::DevBuf<int> dev_distance = host::DevBuf<int>(G.numVertices);
  dev_distance << distance;
  ds.distance = dev_distance;

  std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
  parent[startVertex] = 0;
  host::DevBuf<int> dev_parent = host::DevBuf<int>(G.numVertices);
  dev_parent << parent;
  ds.parent = dev_parent;

  host::DevBuf<int> dev_adjacencyList = host::DevBuf<int>(G.numEdges);
  dev_adjacencyList << G.adjacencyList;
  ds.adjacencyList = dev_adjacencyList;

  host::DevBuf<int> dev_edgesOffset = host::DevBuf<int>(G.numVertices);
  dev_edgesOffset << G.edgesOffset;
  ds.edgesOffset = dev_edgesOffset;

  host::DevBuf<int> dev_edgesSize = host::DevBuf<int>(G.numVertices);
  dev_edgesSize << G.edgesSize;
  ds.edgesSize = dev_edgesSize;

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