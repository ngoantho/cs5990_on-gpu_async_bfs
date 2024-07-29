#include "harmonize.git/harmonize/cpp/harmonize.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace util;

typedef struct {
  int id;
  size_t edge_count;
  size_t edge_offset;
  unsigned int depth; // compare depth of this node and incoming depth
} Node;

// state that will be stored per program instance and accessible by all work
// groups immutable, but can contain references and pointers to non-const data
struct MyDeviceState {
  Node* node_arr;
  int node_count;
  int* edge_arr;
  int root_node;
  bool verbose;
  iter::AtomicIter<unsigned int>* iterator;
};

struct MyProgramOp {
  using Type = void (*)(Node* node, unsigned int current_depth, Node* parent);

  template <typename PROGRAM>
  __device__ static void eval(PROGRAM prog, Node *node, unsigned int current_depth, Node* parent) {
    int this_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int orig_depth = node->depth;

    // if this node is already visited, then skip it
    if (atomicMin(&node->depth, current_depth) <= current_depth) {
      return; // base case
    }

    if (prog.device.verbose) {
      if (parent == nullptr) printf("[%d] node root->%d, depth %u->%u\n", this_id, node->id, orig_depth, node->depth);
      else printf("[%d] node %d->%d, depth %u->%u\n", this_id, parent->id, node->id, orig_depth, node->depth);
    }

    for (int i = 0; i < node->edge_count; i++) {
      int edge_node_id = prog.device.edge_arr[node->edge_offset + i];
      Node& edge_node = prog.device.node_arr[edge_node_id];
      prog.template async<MyProgramOp>(&edge_node, current_depth + 1, node);
    }
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
      prog.template async<MyProgramOp>(&prog.device.node_arr[prog.device.root_node], 0, nullptr);
    }

    return false;
  }
};

using ProgType = AsyncProgram<MyProgramSpec>;

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);

  // arguments
  unsigned int batch_count = args["batch_count"] | 1;
  unsigned int run_count = args["run_count"] | 1;
  unsigned int arena_size = args["arena_size"] | 0x10000;
  std::string file_str = args.get_flag_str((char *)"file");

  // if flag is present, then true, else false
  bool directed = args["directed"];

  // init DeviceState
  MyDeviceState ds;
  ds.node_count = 0;
  ds.root_node = args["root"]; // int
  ds.verbose = args["verbose"]; // bool

  if (ds.verbose) {
    std::cout << "group count: " << batch_count << std::endl;
    std::cout << "cycle count: " << run_count << std::endl;
    std::cout << "arena size: " << arena_size << std::endl;
    std::cout << "parsing " << file_str << std::endl;
  }

  iter::AtomicIter<unsigned int> host_iter(0, 1);
  host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
  iterator << host_iter;
  ds.iterator = iterator;

  std::vector<Node> nodes;
  std::map<int, std::vector<int>> adjacency_graph;

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
    } else if (line_idx == 1) {
      std::string token;
      std::stringstream ss(line);

      // parse node count
      getline(ss, token, ' ');
      ds.node_count = std::stoi(token) + 1;
      if (ds.verbose) {
        std::cout << "loading " << token << " nodes" << std::endl;
      }

      nodes = std::vector<Node>(
        ds.node_count, // 
        {.edge_count = 0, .edge_offset = 0, .depth = 0xFFFFFFFF}
      );

      for (size_t i = 0; i < nodes.size(); i++) {
        nodes.at(i).id = i;
      }
    } else {
      int node_id, edge;
      std::string token;
      std::stringstream ss(line);

      // parse node
      ss >> node_id;

      // parse edge
      ss >> edge;
      adjacency_graph[node_id].push_back(edge);

      if (!directed) {
        adjacency_graph[edge].push_back(node_id);
      }
    }
  }

  // finally close file
  file.close();

  // single edge array
  std::vector<int> edges;

  for (std::map<int, std::vector<int>>::iterator it = adjacency_graph.begin(); it != adjacency_graph.end(); it++) {
    size_t offset = edges.size(); // size before adding edges

    for (auto &&edge : it->second) {
      edges.push_back(edge);
    }

    Node node = {.id = it->first,
                 .edge_count = it->second.size(),
                 .edge_offset = offset,
                 .depth = 0xFFFFFFFF};
    nodes.at(node.id) = node;
  }

  host::DevBuf<int> dev_edges(edges.size());
  dev_edges << edges;
  ds.edge_arr = dev_edges;

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
    exec<ProgType>(instance, batch_count, run_count);
    cudaDeviceSynchronize();
    host::check_error();
  } while (!instance.complete());
}