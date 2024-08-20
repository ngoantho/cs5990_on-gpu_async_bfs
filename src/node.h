#ifndef NODE
#define NODE

typedef struct {
  int id;
  size_t edge_count;
  size_t edge_offset;
  unsigned int depth;
  unsigned int visited;
} Node;

#endif