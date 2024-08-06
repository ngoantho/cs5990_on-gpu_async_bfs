#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include "harmonize.git/harmonize/cpp/util/cli.h"

#include <iostream>
#include <fstream>

using namespace util;

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);

  char* file_str = args.get_flag_str((char*)"file");
  if (file_str == nullptr) {
    std::cerr << "no value provided for -file" << std::endl;
    std::exit(1);
  }

  std::ifstream file(file_str);
  if (!file.is_open()) {
    std::cerr << "unable to open " << file_str << std::endl;
    std::exit(1);
  }

  file.close();
}