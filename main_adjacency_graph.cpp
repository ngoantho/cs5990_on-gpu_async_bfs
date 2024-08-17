#include "adjacency_graph.h"
#include "cli_argset.h"
#include <sstream>
#include <iterator>

std::string vec_to_string(std::vector<int>& vec) {
  std::ostringstream oss;
  std::copy(vec.begin(), vec.end()-1, std::ostream_iterator<int>(oss, " "));
  oss << vec.back();
  return oss.str();
}

int main(int argc, char* argv[]) {
  cli::ArgSet args(argc, argv);
  char* file_str = args.get_flag_str((char*)"file");
  int node_count = 0;
  bool directed = args["directed"];
  AdjacencyGraph graph = AdjacencyGraph(file_str, node_count, directed, false);

  char* comments = args.get_flag_str((char*)"comments");
  bool skip_first_line = args["skip_first_line"];
  char delimiter = args["delimiter"];
  graph.handle_parsing(nullptr, true, ',', true);
  
  graph.file.close();
}