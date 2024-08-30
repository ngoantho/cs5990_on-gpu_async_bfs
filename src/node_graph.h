#ifndef NODE_GRAPH
#define NODE_GRAPH

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "node.h"
struct NodeGraph {
  std::vector<Node> nodes;
  std::vector<int> edges;

  NodeGraph(std::map<int, std::vector<int>>& adjacency_graph, bool verbose=false) {
    // map keys are sorted ascendingly by default
    int node_count = adjacency_graph.rbegin()->first + 1;
    
    nodes = std::vector<Node>(node_count);
    edges = std::vector<int>(node_count);

    for (int i = 0; i < node_count; i++) {
      nodes.at(i) = Node(i);
    }

    for (std::map<int, std::vector<int>>::iterator it = adjacency_graph.begin(); it != adjacency_graph.end(); it++) {
      size_t offset = edges.size(); // size before adding edges

      for (int edge : it->second) {
        edges.push_back(edge);
      }

      nodes.at(it->first).edge_count = it->second.size();
      nodes.at(it->first).edge_offset = offset;
    }

    int max_edge = *std::max_element(edges.begin(), edges.end());
    if (verbose) {
      std::cout << "max edge: " << max_edge << std::endl;
      std::cout << "node count: " << node_count << std::endl;
    }
    if (max_edge >= node_count) {
      nodes.resize(max_edge + 1); // edges may reference nodes not added yet
      if (verbose) {
        std::cout << "resizing graph to: " << nodes.size() << std::endl;
      }

      for (int i = node_count; i < nodes.size(); i++) {
        nodes.at(i) = Node(i);
      }
    }
  }
};

#endif