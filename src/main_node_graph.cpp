#include "cli_argset.h"
#include "file_parser.h"
#include "node_graph.h"

int main(int argc, char *argv[]) {
  cli::ArgSet args(argc, argv);
  bool directed = args["directed"];
  char *file_str = args.get_flag_str((char *)"file");

  FileParser file_parser;
  if (file_parser.known_extension(file_str)) 
    file_parser.parse_extension(file_str); 
  else 
    file_parser.parse_arguments(args);

  std::map<int, std::vector<int>>& adjgraph = file_parser.parse_file(file_str, directed);
  NodeGraph node_graph(adjgraph, true);
  
  for (Node& node : node_graph.nodes) {
    std::cout << node.id << ": edge_count=" << node.edge_count << std::endl;
    size_t offset = node.edge_offset;
    for (size_t i = 0; i < node.edge_count; i++) {
      int edge_node_id = node_graph.edges[offset + i];
      std::cout << "- " << edge_node_id << std::endl;
    }
    std::cout << std::endl;
  }
}