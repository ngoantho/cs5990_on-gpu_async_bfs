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

  NodeGraph(std::map<int, std::vector<int>>& adjacency_graph, int node_count): nodes(node_count), edges() {
    for (size_t i = 0; i < nodes.size(); i++) {
      nodes.at(i).id = i;
      nodes.at(i).depth = 0xFFFFFFFF;
      nodes.at(i).visited = 0;
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