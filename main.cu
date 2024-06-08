#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "harmonize.git/harmonize/cpp/harmonize.h"
using namespace util;

// the state that will be stored per program instance and accessible by all work groups
struct DeviceState
{
};

struct ProgramSpec
{
  static const size_t STASH_SIZE = 16;
  static const size_t FRAME_SIZE = 8191;
  static const size_t POOL_SIZE = 8191;

  template <typename PROGRAM>
  __device__ static void initialize(PROGRAM prog)
  {
  }

  template <typename PROGRAM>
  __device__ static void finalize(PROGRAM prog)
  {
  }

  template <typename PROGRAM>
  __device__ static bool make_work(PROGRAM prog)
  {
    
  }
};

#define NUM_NODES 5

typedef struct {
  int start;
  int length;
} Node;

__global__ void naive_bfs(Node *nodes, int *edges, bool *frontier, bool *visited, int *cost, bool *done)
{

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id > NUM_NODES)
    *done = false;

  if (frontier[id] == true && visited[id] == false)
  {
    printf("%d ", id); // This printf gives the order of vertices in BFS
    frontier[id] = false;
    visited[id] = true;
    __syncthreads();
    int k = 0;
    int i;
    int start = nodes[id].start;
    int end = start + nodes[id].length;
    for (int i = start; i < end; i++)
    {
      int nid = edges[i];

      if (visited[nid] == false)
      {
        cost[nid] = cost[id] + 1;
        frontier[nid] = true;
        *done = false;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  cli::ArgSet args(argc, argv);

  int num_blocks = args["num_blocks"] | 1;
  int threads = args["threads"] | 5;

  Node nodes[NUM_NODES];
  int edges[NUM_NODES];

  nodes[0].start = 0;
  nodes[0].length = 2;

  nodes[1].start = 2;
  nodes[1].length = 1;

  nodes[2].start = 3;
  nodes[2].length = 1;

  nodes[3].start = 4;
  nodes[3].length = 1;

  nodes[4].start = 5;
  nodes[4].length = 0;

  edges[0] = 1;
  edges[1] = 2;
  edges[2] = 4;
  edges[3] = 3;
  edges[4] = 4;

  bool frontier[NUM_NODES] = {false};
  bool visited[NUM_NODES] = {false};
  int cost[NUM_NODES] = {0};

  int source = 0;
  frontier[source] = true;

  Node* d_node;
  cudaMalloc((void**)&d_node, sizeof(Node)*NUM_NODES);
  cudaMemcpy(d_node, nodes, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

  int *d_edge;
  cudaMalloc((void **)&d_edge, sizeof(Node) * NUM_NODES);
  cudaMemcpy(d_edge, edges, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

  bool *d_frontier;
  cudaMalloc((void **)&d_frontier, sizeof(bool) * NUM_NODES);
  cudaMemcpy(d_frontier, frontier, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

  bool *d_visited;
  cudaMalloc((void **)&d_visited, sizeof(bool) * NUM_NODES);
  cudaMemcpy(d_visited, visited, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

  int *d_cost;
  cudaMalloc((void **)&d_cost, sizeof(int) * NUM_NODES);
  cudaMemcpy(d_cost, cost, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

  bool done;
  bool *d_done;
  cudaMalloc((void **)&d_done, sizeof(bool));
  int count = 0;

  printf("Order: \n\n");
  do {
    count++;
    done = true;
    cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
    naive_bfs<<<num_blocks, threads>>>(d_node, d_edge, d_frontier, d_visited, d_cost, d_done);
    cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (!done);

  cudaMemcpy(cost, d_cost, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);
  printf("number of times the kernel is called: %d\n", count);

  printf("\nCost: ");
  for (int i = 0; i < NUM_NODES; i++)
    printf("%d    ", cost[i]);
  printf("\n");
}