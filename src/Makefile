all: file_parser queue harmonize main

clean:
	rm -f *.exe

queue: main_queue.cpp
	g++ -g -std=c++17 -o queue.exe main_queue.cpp

harmonize: main_harmonize.cu
	nvcc -g -G -std=c++17 -o harmonize.exe main_harmonize.cu

file_parser: file_parser.h main_file_parser.cpp
	g++ -g -std=c++17 -o file_parser.exe main_file_parser.cpp

node_graph: node_graph.h main_node_graph.cpp
	g++ -g -std=c++17 -o node_graph.exe main_node_graph.cpp

main: main_queue.cpp main_harmonize.cu main.cpp
	nvcc -g -G -x cu -std=c++17 -o main.exe main.cpp