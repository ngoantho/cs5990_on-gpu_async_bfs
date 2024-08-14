all: main_harmonize.exe main_queue.exe

clean:
	rm -f *.exe

main_queue.exe: main_queue.cpp
	g++ -std=c++17 $(GCCFLAGS) -o main_queue.exe main_queue.cpp

main_harmonize.exe: main_harmonize.cu
	nvcc -std=c++17 -lineinfo $(GCCFLAGS) -o main_harmonize.exe main_harmonize.cu
