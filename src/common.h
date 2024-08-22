#ifndef COMMON
#define COMMON

#include <vector>
#include <string>
#include <iostream>
#include "node.h"
#include "cli_argset.h"

void common_for(std::ostream& dest, std::vector<Node>& nodes) {
  for (Node &i : nodes) {
    dest << i.id << ": depth=" << i.depth << ", visited=" << i.visited << std::endl;
  }
}

void common_output(cli::ArgSet& args, float runtime, std::vector<Node>& nodes, std::string type) {
  const char* output = args.get_flag_str((char*)"output");
  if (output == nullptr) {
    output = std::string("runtime").c_str();
  }

  if (std::string(output) == "runtime") {
    std::cout << "Runtime: " << runtime << "ms" << std::endl;
  } else if (std::string(output) == "state") {
    common_for(std::cout, nodes);
  } else if (std::string(output) == "write") {
    std::string filename = "./out/"+type+".txt";
    std::cout << "writing to: " << filename << std::endl;
    std::ofstream file(filename);
    common_for(file, nodes); // ios_base -> ios -> ostream -> ofstream
    file.close();
  } else {
    std::cerr << "unknown output value: " << std::string(output) << std::endl;
    std::exit(1);
  }
}

#endif