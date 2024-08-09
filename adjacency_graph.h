#ifndef ADJACENCY_GRAPH
#define ADJACENCY_GRAPH

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

struct AdjacencyGraph {
  std::map<int, std::vector<int>> data;

  AdjacencyGraph(char* file_str, int& node_count, bool directed): data() {
    std::ifstream file(file_str);
    if (!file.is_open()) {
      std::cerr << "unable to open " << file_str << std::endl;
      std::exit(1);
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
        node_count = std::stoi(token) + 1; // need +1 for nodes n
      } else {
        int node_id, edge;
        std::stringstream ss(line);

        // parse node
        ss >> node_id;

        // parse edge
        ss >> edge;
        data[node_id].push_back(edge);

        if (!directed) {
          data[edge].push_back(node_id);
        }
      }
    }

    // finally close file
    file.close();
  }
};

#endif