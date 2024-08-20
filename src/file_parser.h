#ifndef FILE_PARSER
#define FILE_PARSER

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>

#include <filesystem>
namespace fs = std::filesystem;

#include "cli_argset.h"

bool string_contains(std::string str, std::string sub) {
  return str.find(sub) != std::string::npos;
}

struct FileParser {
  std::map<int, std::vector<int>> adjacency_graph;
  std::vector<int> graph_keys;
  int node_count;

  const char* comments;
  bool skip_first_line;
  const char* delimiter;

  FileParser()
      : adjacency_graph(), node_count(0),
        comments(nullptr), skip_first_line(false), delimiter(nullptr) 
  {}

  fs::path get_extension(char* file_str) {
    if (file_str == nullptr) {
      std::cerr << "file string null" << std::endl;
      std::exit(1);
    }
    fs::path path(file_str);
    return path.extension();
  }

  bool known_extension(char* file_str) {
    fs::path extension = get_extension(file_str);
    return extension == ".mtx" || extension == ".csv";
  }

  void parse_extension(char* file_str) {
    fs::path extension = get_extension(file_str);
    if (extension == ".mtx") {
      comments = "%%";
      skip_first_line = true;
      delimiter = " ";
    } else if (extension == ".csv") {
      comments = nullptr;
      skip_first_line = true;
      delimiter = ",";
    }
  }

  void parse_arguments(cli::ArgSet& args, bool verbose=false) {
    comments = args.get_flag_str((char*)"comment");
    if (verbose && comments != nullptr) std::cout << "comment: " << comments << std::endl;
    else if (verbose && comments == nullptr) std::cout << "comment: null" << std::endl;

    skip_first_line = args["skip_first_line"];
    if (verbose) std::cout << "skip first line: " << skip_first_line << std::endl;

    delimiter = args.get_flag_str((char*)"delimiter");
    if (delimiter == nullptr) {
      std::cout << "no value provided for -delimiter" << std::endl;
      std::exit(1);
    } else if (string_contains(delimiter, "tabs") || string_contains(delimiter, "/t")) {
      delimiter = "	";
    } else if (std::string(delimiter).length() > 1) {
      std::cout << "invalid delimiter" << std::endl;
      std::exit(1);
    } else if (verbose) std::cout << "delimiter: " << delimiter << std::endl;
  }

  std::map<int, std::vector<int>>& parse_file(char* file_str, bool directed, bool verbose=false) {
    std::ifstream file(file_str);
    if (!file.is_open()) {
      std::cerr << "unable to open " << file_str << std::endl;
      std::exit(1);
    }

    std::string line;
    unsigned int line_idx = 0;
    bool skipped_no_comments = false, skipped_first_line = false;
    while (std::getline(file, line)) {
      line_idx++;
      if (comments == nullptr && !skipped_no_comments) {
        if (verbose) std::cout << "skipping comments" << std::endl;
        skipped_no_comments = true;
      } else if (comments != nullptr && string_contains(line, comments)) {
        if (verbose) std::cout << "skipping " << line << std::endl;
        continue;
      } else if (skip_first_line && !skipped_first_line) {
        if (verbose) std::cout << "skipping first line" << std::endl;
        skipped_first_line = true;
      } else {
        std::stringstream ss(line);
        std::string token;
        int node, edge;

        getline(ss, token, *delimiter);
        node = std::stoi(token);

        getline(ss, token, *delimiter);
        edge = std::stoi(token);

        graph_keys.push_back(node);
        adjacency_graph[node].push_back(edge);
        
        // directed graphs point in one direction
        if (!directed) adjacency_graph[edge].push_back(node);
      }
    }

    file.close();
    node_count = *std::max_element(graph_keys.begin(), graph_keys.end());
    return adjacency_graph;
  }
};

#endif