all: test_fire.exe

clean:
	rm -f *.exe

test_fire.exe: test_fire.cpp
	nvcc -x cu -std=c++17 -o test_fire.exe test_fire.cu -I./harmonize.git/harmonize/cpp