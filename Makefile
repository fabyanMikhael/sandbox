CXX=g++
NVCC=nvcc
TARGET=target/main
OBJ_DIR=target
SRC_DIR=src
CXX_SOURCES=$(wildcard $(SRC_DIR)/**/*.cpp $(SRC_DIR)/*.cpp)
CU_SOURCES=$(wildcard $(SRC_DIR)/**/*.cu $(SRC_DIR)/*.cu)
CXX_OBJECTS=$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CXX_SOURCES))
CU_OBJECTS=$(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SOURCES))

all: $(TARGET)

$(TARGET): $(CXX_OBJECTS) $(CU_OBJECTS)
	$(CXX) -L/usr/local/cuda/lib64 -o $@ $^ -lSDL2 -lcudart 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) -c -o $@ $<

clean: 
	@rm -rf $(OBJ_DIR) $(TARGET)

run: $(TARGET)
	@./$(TARGET)