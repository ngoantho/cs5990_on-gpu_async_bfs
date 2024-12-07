all: file_parser queue harmonize main

clean:
	rm -f *.exe

queue: src/main_queue.cpp
	g++ -g -std=c++17 -o queue.exe src/main_queue.cpp

harmonize: src/main_harmonize.cu
	nvcc -g -G -std=c++17 -o harmonize.exe src/main_harmonize.cu

file_parser: src/file_parser.h src/main_file_parser.cpp
	g++ -g -std=c++17 -o file_parser.exe src/main_file_parser.cpp

node_graph: src/node_graph.h src/main_node_graph.cpp
	g++ -g -std=c++17 -o node_graph.exe src/main_node_graph.cpp

main: src/main_queue.cpp src/main_harmonize.cu src/main.cpp
	nvcc -g -G -x cu -std=c++17 -o main.exe src/main.cpp