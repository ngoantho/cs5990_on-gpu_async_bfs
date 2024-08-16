#ifndef COMMON
#define COMMON

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include "harmonize.git/harmonize/cpp/util/cli.h"
using namespace util;

#include <vector>
#include <string>
#include <iostream>
#include "node.h"

void common_output(cli::ArgSet& args, float runtime, std::vector<Node>& nodes, std::string type) {
  const char* output = args.get_flag_str((char*)"output");
  if (output == nullptr) {
    output = std::string("runtime").c_str();
  }

  auto common_for = [nodes](std::ostream& dest) {
    for (auto &&i : nodes) {
      dest << i.id << ": depth=" << i.depth << ", visited=" << i.visited << std::endl;
    }
  };

  if (std::string(output) == "runtime") {
    std::cout << "Runtime: " << runtime << "ms" << std::endl;
  } else if (std::string(output) == "state") {
    common_for(std::cout);
  } else if (std::string(output) == "write") {
    std::string filename = "/tmp/"+type+".txt";
    std::cout << "writing to: " << filename << std::endl;
    std::ofstream file(filename);
    common_for(file); // ios_base <- ios <- ostream <- ofstream
  } else {
    std::cerr << "unknown output value: " << std::string(output) << std::endl;
    std::exit(1);
  }
}

#endif