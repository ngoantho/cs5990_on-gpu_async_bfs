all: main graph

clean:
	rm -f *.exe

main: main.cu graph.cpp
	nvcc -std=c++17 -o main.exe main.cu graph.cpp -I./harmonize.git/harmonize/cpp

graph: graph.cpp
	g++ -DGRAPH_MAIN -std=c++17 -o graph.exe graph.cpp