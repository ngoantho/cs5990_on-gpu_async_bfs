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

  NodeGraph(std::map<int, std::vector<int>>& adjacency_graph, int node_count): nodes(node_count+1), edges(node_count+1) {
    // Node zero;
    // zero.id = 0;
    // zero.depth = 0xFFFFFFFF;
    // zero.visited = 0;
    // zero.edge_count = 0;
    // zero.edge_offset = 0;
    // nodes.push_back(zero);

    for (int i = 0; i < node_count+1; i++) {
      Node node;
      node.id = i;
      node.depth = 0xFFFFFFFF;
      node.visited = 0;
      node.edge_count = 0;
      nodes.at(i) = node;
    }

    for (std::map<int, std::vector<int>>::iterator it = adjacency_graph.begin(); it != adjacency_graph.end(); it++) {
      size_t offset = edges.size(); // size before adding edges

      for (int edge : it->second) {
        edges.push_back(edge);
      }

      nodes.at(it->first).edge_count = it->second.size();
      nodes.at(it->first).edge_offset = offset;
    }
  }
};

#endif