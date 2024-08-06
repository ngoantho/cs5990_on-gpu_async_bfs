all: main.exe graph.exe

clean:
	rm -f *.exe

graph.exe: graph.cpp
	nvcc -std=c++17 -o graph.exe graph.cpp

main.exe: main.cu
	nvcc -std=c++17 -o main.exe main.cu
