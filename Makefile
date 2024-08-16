all: main.exe

clean:
	rm -f *.exe

queue.exe: main_queue.cpp
	g++ -std=c++17 -o queue.exe main_queue.cpp

harmonize.exe: main_harmonize.cu
	nvcc -std=c++17 -lineinfo -o harmonize.exe main_harmonize.cu

main.exe: main_queue.cpp main_harmonize.cu main.cpp
	nvcc -x cu -std=c++17 -lineinfo -o main.exe main.cpp