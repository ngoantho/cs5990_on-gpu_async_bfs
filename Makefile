all: main.exe

clean:
	rm -f *.exe

queue.exe: main_queue.cpp
	g++ -std=c++17 -o queue.exe main_queue.cpp

harmonize.exe: main_harmonize.cu
	nvcc -std=c++17 -lineinfo -o harmonize.exe main_harmonize.cu

adjacency_graph.exe: adjacency_graph.h main_adjacency_graph.cpp
	g++ -std=c++17 -o adjacency_graph.exe main_adjacency_graph.cpp

main.exe: main_queue.cpp main_harmonize.cu main.cpp
	nvcc -x cu -std=c++17 -lineinfo -o main.exe main.cpp