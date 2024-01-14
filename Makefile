CXX=g++
NVCC=nvcc

all: main

main: main.o renderer.o particles.o
	${CXX} -L/usr/local/cuda/lib64 -o target/main target/main.o target/renderer.o target/gpu.o target/particles.o -lSDL2 -lcudart 

main.o: src/main.cpp
	${CXX} -c -o target/main.o src/main.cpp

renderer.o: src/renderer/renderer.cpp src/renderer/renderer.h gpu.o
	${CXX} -c -o target/renderer.o src/renderer/renderer.cpp

gpu.o: src/renderer/gpu.cu src/renderer/gpu.h
	nvcc -c -o target/gpu.o src/renderer/gpu.cu

particles.o: src/particles/particles.cu src/particles/particles.h
	nvcc -c -o target/particles.o src/particles/particles.cu

clean: 
	rm -f target/*.o main

run: main
	./target/main