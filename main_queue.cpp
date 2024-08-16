#include <iostream>
#include <fstream>
#include <queue>

#include <chrono>
using namespace std::chrono;

#include "adjacency_graph.h"
#include "node_graph.h"
#include "common.h"

int main_queue(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);

  char* file_str = args.get_flag_str((char*)"file");
  if (file_str == nullptr) {
    std::cerr << "no value provided for -file" << std::endl;
    std::exit(1);
  }

  // if flag is present, then true, else false
  bool directed = args["directed"];

  // modified during construction of the graphs
  int node_count;

  AdjacencyGraph adjacency_graph(file_str, node_count, directed);
  NodeGraph node_graph(adjacency_graph.data, node_count);

  int root_node_id = args["root"];
  Node* root_node = &node_graph.nodes[root_node_id];

  std::queue<Node*> queue;
  root_node->depth = 0;
  queue.push(root_node);

  auto start = high_resolution_clock::now();

  while (!queue.empty()) {
    Node* node = queue.front();
    queue.pop();

    if (node->visited == 1) continue;
    else node->visited = 1;

    for (int i = 0; i < node->edge_count; i++) {
      int edge_node_id = node_graph.edges[node->edge_offset + i];
      Node& edge_node = node_graph.nodes[edge_node_id];
      edge_node.depth = std::min(edge_node.depth, node->depth+1);
      queue.push(&edge_node);
    }
  }

  auto stop = high_resolution_clock::now();
  duration<float, std::milli> ms = stop - start;

  // normalize output format
  common_output(args, ms.count(), node_graph.nodes, "queue");

  return 0;
}

#ifndef MAIN
int main(int argc, char* argv[]) {
  return main_queue(argc, argv);
}
#endif