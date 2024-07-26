#include "harmonize.git/harmonize/cpp/harmonize.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <map> 
using namespace util;

typedef struct {
  int id;
  size_t edge_count; // edges that branch off
  int* edge_arr; // idx of nodes
  int visited; // base case: node is visited
  unsigned int depth; // compare depth of this node and incoming depth
} Node;

// state that will be stored per program instance and accessible by all work
// groups immutable, but can contain references and pointers to non-const data
struct MyDeviceState {
  Node* node_arr;
  int node_count;
  iter::AtomicIter<unsigned int>* iterator;
};

struct MyProgramOp {
  using Type = void (*)(Node* node, unsigned int current_depth);

  template <typename PROGRAM>
  __device__ static void eval(PROGRAM prog, Node* node, unsigned int current_depth) {
    /* TODO parse node
    int this_id = blockIdx.x * blockDim.x + threadIdx.x;

    // if this node is already visited, then skip it
    if (atomicMin(&node->depth, current_depth) <= current_depth) {
      printf("[%d] node %d visited already: depth=%u, incoming=%u\n", this_id, node->id, node->depth, current_depth);
      return; // base case
    }

    printf("[%d] node %d: depth=%u\n", this_id, node->id, node->depth);

    for (int i = 0; i < node->edge_count; i++) {
      int edge_id = node->edge_arr[i];
      Node& edge_node = prog.device.node_arr[edge_id];
      prog.template async<MyProgramOp>(&edge_node, current_depth + 1);
    }
    */
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
    /* TODO init traversal
    unsigned int index;
    if (prog.device.iterator->step(index)) {
      prog.template async<MyProgramOp>(&prog.device.node_arr[0], 0);
    }
    */
    Node node = prog.device.node_arr[28];
    printf("%d\n", node.id);
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
  std::string file_str = args.get_flag_str("file");
  std::cout << "parsing " << file_str << "..." << std::endl;

  // init DeviceState
  MyDeviceState ds;
  ds.node_count = 0;

  iter::AtomicIter<unsigned int> host_iter(0, 1);
	host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
	iterator << host_iter;
	ds.iterator = iterator;

  std::vector<Node> nodes;
  std::map<std::string, std::vector<int>> adjacency_graph;

  std::ifstream file(file_str);
  if (!file.is_open()) {
    std::cerr << "unable to open " << file_str << std::endl;
    return 1;
  }

  std::string line;
  unsigned int line_idx = 0;
  while (std::getline(file, line)) {
    line_idx++;
    if (line.substr(0, 2) == "%%") {
      line_idx--; // trigger line_idx == 1
    }
    else if (line_idx == 1) {
      std::string token;
      std::stringstream ss(line);

      // parse node count
      getline(ss, token, ' ');
      ds.node_count = std::stoi(token) + 1;
      nodes = std::vector<Node>(ds.node_count);
    }
    else {
      std::string token, node;
      std::stringstream ss(line);

      // parse node
      getline(ss, token, ' ');
      node = token;

      // parse edge
      getline(ss, token, ' ');
      adjacency_graph[node].push_back(std::stoi(token));
    }
  }

  // finally close file
  file.close();

  for(std::map<std::string, std::vector<int>>::iterator it = adjacency_graph.begin(); it != adjacency_graph.end(); it++) {
    std::string id = it->first;
    std::vector<int> edges = it->second;
    
    host::DevBuf<int> dev_edges(edges.size());
    dev_edges << edges;

    Node node = {
      .id = std::stoi(id),
      .edge_count = edges.size(),
      .edge_arr = dev_edges,
      .visited = 0,
      .depth = 0xFFFFFFFF
    };
    nodes.at(node.id) = node;
  }

  host::DevBuf<Node> dev_nodes(ds.node_count);
  dev_nodes << nodes;
  ds.node_arr = dev_nodes;

  if (ds.node_count == 0) {
    std::cerr << "error: node count = 0" << std::endl;
    return 0;
  }
  
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