#ifndef ADJACENCY_GRAPH
#define ADJACENCY_GRAPH

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include <filesystem>
namespace fs = std::filesystem;

struct AdjacencyGraph {
  std::map<int, std::vector<int>> data;
  std::ifstream file;
  fs::path path;
  int &node_count;
  bool directed;

  AdjacencyGraph(char *_file_str, int &_node_count, bool _directed, bool parse_file=true)
      : data(), file(_file_str), path(_file_str), node_count(_node_count),
        directed(_directed) {
    if (!file.is_open()) {
      std::cerr << "unable to open " << path << std::endl;
      std::exit(1);
    }

    if (parse_file) {
      handle_extension(path.extension());
      file.close();
    }
  }

  void handle_parsing(const char *comments, bool skip_first_line, char delimiter, bool verbose=false) {
    std::string line;
    unsigned int line_idx = 0;
    bool skipped_no_comments = false, skipped_first_line = false;
    while (std::getline(file, line)) {
      line_idx++;
      if (comments == nullptr && !skipped_no_comments) {
        if (verbose) std::cout << "skipping no comments" << std::endl;
        skipped_no_comments = true;
      } else if (comments != nullptr && line.find(comments) != std::string::npos) {
        if (verbose) std::cout << "skipping " << line << std::endl;
        continue;
      } else if (skip_first_line && !skipped_first_line) {
        if (verbose) std::cout << "skipping first line" << std::endl;
        skipped_first_line = true;
      } else {
        break;
      }
    }
  }

  void handle_mtx() {
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
  }

  void handle_extension(fs::path extension) {
    std::cout << "detected extension: " << extension << std::endl;
    if (extension == ".mtx") {
      handle_parsing("%%", true, ' ');
    } else {
      std::cerr << "unknown extension: " << extension << std::endl;
      std::exit(1);
    }
  }
};

#endif