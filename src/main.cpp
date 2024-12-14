#define MAIN // disable main_harmonize's main

#include "main_harmonize.cu"
#include "main_queue.cpp"

#include <algorithm> // std::find

int main(int argc, char *argv[]) {
  std::cout << "--- harmonize ---" << std::endl;
  int res_harmonize = main_harmonize(argc, argv);
  if (res_harmonize != 0) {
    std::exit(res_harmonize);
  }
  std::cout << "-----------------" << std::endl;

  std::cout << "--- queue ---" << std::endl;
  int res_queue = main_queue(argc, argv);
  if (res_queue != 0) {
    std::exit(res_queue);
  }
  std::cout << "-------------" << std::endl;

  return 0;
}
