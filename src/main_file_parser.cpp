#include "cli_argset.h"
#include "file_parser.h"
#include <iterator>
#include <sstream>
#include <vector>

std::string vec_to_string(std::vector<int> &vec) {
  if (vec.size() > 0) {
    std::ostringstream oss;
    std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<int>(oss, " "));
    oss << vec.back();
    return oss.str();
  } else {
    return "";
  }
}

int max_key(std::map<int, std::vector<int>>& graph) {
  std::vector<int> vec;
  for (auto &&pair : graph)
  {
    vec.push_back(pair.first);
  }
  return *std::max_element(vec.begin(), vec.end());
}

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);
  bool directed = args["directed"];
  char *file_str = args.get_flag_str((char *)"file");

  FileParser file_parser;
  // if (file_parser.known_extension(file_str))
  // file_parser.parse_extension(file_str); else
  // file_parser.parse_arguments(args);
  file_parser.parse_arguments(args);

  std::map<int, std::vector<int>>& graph = file_parser.parse_file(file_str, directed);

  std::cout << "max node - graph.rbegin: " << graph.rbegin()->first << std::endl;
  std::cout << "max node - max element: " << max_key(graph) << std::endl;
}