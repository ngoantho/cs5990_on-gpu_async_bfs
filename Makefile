all: run

clean:
	rm -f *.exe

run: main.exe
	./main.exe -file $(file) -root $(root)

main.exe: main.cu
	nvcc -std=c++17 -o main.exe main.cu
