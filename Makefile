all: main.exe

clean:
	rm -f *.exe

main.exe: main.cu
	nvcc -std=c++17 -o main.exe main.cu
