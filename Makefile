all: run

clean:
	rm -f *.exe

run: main.exe
	./main.exe

main.exe: main.cu
	nvcc -std=c++17 -o main.exe main.cu
