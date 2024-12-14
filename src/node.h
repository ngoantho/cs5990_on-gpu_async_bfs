#ifndef NODE
#define NODE

struct Node {
  int id;
  size_t edge_count;
  size_t edge_offset;
  unsigned int depth;
  unsigned int visited;

  Node() {}

  Node(int _id) : id(_id), edge_count(0), depth(0xFFFFFFFF), visited(0) {}
};

#endif