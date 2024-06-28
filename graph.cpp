#include "graph.h"
#include <ctime>
#include <iostream>

Graph::Graph(int numVertices, int numEdges)
{
  std::cout << "numVertices: " << numVertices << std::endl;
  std::cout << "numEdges: " << numEdges << std::endl;
  srand(12345);

  std::vector<std::vector<int>> adjacencyList(numVertices);
  for (int i = 0; i < numEdges; i++)
  {
    int vertex = rand() % numVertices;
    int edge = rand() % numVertices;
    adjacencyList[vertex].push_back(edge);
    adjacencyList[edge].push_back(vertex);
    std::cout << "adjacencyList vertex:[" << vertex << "] = edge:" << edge << std::endl;
    std::cout << "adjacencyList edge:[" << edge << "] = vertex:" << vertex << std::endl;
  }
  
  for (int i = 0; i < numVertices; i++)
  {
    this->edgesOffset.push_back(this->adjacencyList.size());
    this->edgesSize.push_back(adjacencyList[i].size());
    for (auto& edge: adjacencyList[i]) {
      this->adjacencyList.push_back(edge);
    }
  }
  
  this->numVertices = numVertices;
  this->numEdges = this->adjacencyList.size();
}

#ifdef GRAPH_MAIN
int main(int argc, char* argv[]) {
  int numVertices = atoi(argv[1]);
  int numEdges = atoi(argv[2]);
  Graph g(numVertices, numEdges);
  return 0;
}
#endif