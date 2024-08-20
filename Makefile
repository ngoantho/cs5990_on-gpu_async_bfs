all: file_parser.exe queue.exe harmonize.exe main.exe

clean:
	rm -f *.exe

queue.exe: src/main_queue.cpp
	g++ -g -std=c++17 -o queue.exe src/main_queue.cpp

harmonize.exe: src/main_harmonize.cu
	nvcc -std=c++17 -lineinfo -o harmonize.exe src/main_harmonize.cu

file_parser.exe: src/file_parser.h src/main_file_parser.cpp
	g++ -g -std=c++17 -o file_parser.exe src/main_file_parser.cpp

main.exe: src/main_queue.cpp src/main_harmonize.cu src/main.cpp
	nvcc -x cu -std=c++17 -lineinfo -o main.exe src/main.cpp