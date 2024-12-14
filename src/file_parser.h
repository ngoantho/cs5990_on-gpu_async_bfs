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

#include "cli_argset.h"

bool string_contains(std::string str, std::string sub) {
  return str.find(sub) != std::string::npos;
}

struct FileParser {
  std::map<int, std::vector<int>> adjacency_graph;
  bool verbose;

  const char* comments;
  bool skip_first_line;
  const char* delimiter;

  FileParser(bool _verbose=false)
      : adjacency_graph(), verbose(_verbose),
        comments(nullptr), skip_first_line(false), delimiter(nullptr) 
  {}

  std::string get_extension(char* file_str) {
    if (file_str == nullptr) {
      std::cerr << "file string null" << std::endl;
      std::exit(1);
    }
    
    size_t dotPos = std::string(file_str).rfind('.');
    if (dotPos != std::string::npos) {
      return std::string(file_str).substr(dotPos);
    }
    return "";
  }

  bool known_extension(char* file_str) {
    std::string extension = get_extension(file_str);
    return extension == ".mtx" || extension == ".csv";
  }

  void parse_extension(char* file_str) {
    std::string extension = get_extension(file_str);
    if (extension == ".mtx") {
      comments = "%%";
      skip_first_line = true;
      delimiter = " ";
    } else if (extension == ".csv") {
      comments = nullptr;
      skip_first_line = false;
      delimiter = ",";
    }
  }

  void parse_arguments(cli::ArgSet& args) {
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

  std::map<int, std::vector<int>>& parse_file(char* file_str, bool directed) {
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
      } else if (line == "") {
        if (verbose) std::cout << "skipping empty line" << std::endl;
        continue;
      } else {
        std::stringstream ss(line);
        std::string token;
        int node, edge;

        getline(ss, token, *delimiter);
        node = std::stoi(token);

        getline(ss, token, *delimiter);
        edge = std::stoi(token);

        adjacency_graph[node].push_back(edge);
        
        // directed graphs point in one direction
        if (!directed) adjacency_graph[edge].push_back(node);
      }
    }

    file.close();
    return adjacency_graph;
  }
};

#endif