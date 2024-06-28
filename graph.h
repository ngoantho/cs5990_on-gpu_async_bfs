#ifndef BFS_GRAPH
#define BFS_GRAPH

#include <vector>

struct Graph {
  std::vector<int> adjacencyList; // edges
  std::vector<int> edgesOffset; // offset to edge for each vertex
  std::vector<int> edgesSize; // number of edges for each vertex
  int numVertices = 0;
  int numEdges = 0;

  Graph(int numVertices, int numEdges);
};

#endif // BFS_GRAPH