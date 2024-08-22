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

  NodeGraph(std::map<int, std::vector<int>>& adjacency_graph) {
    // map keys are sorted ascendingly by default
    int node_count = adjacency_graph.rbegin()->first + 1;
    
    nodes = std::vector<Node>(node_count);
    edges = std::vector<int>(node_count);

    for (int i = 0; i < node_count; i++) {
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