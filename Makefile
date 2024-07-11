all: main

clean:
	rm -f *.exe

main: main.cu
	nvcc -std=c++17 -o main.exe main.cu -I./harmonize.git/harmonize/cpp
