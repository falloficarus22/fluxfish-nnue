CXX = g++
CXXFLAGS = -O3 -mavx2 -mfma -Wall -std=c++17 -IFathom/src

TARGET = fluxfish_cpp
SRCS = main.cpp nnue_inference.cpp mcts.cpp tablebase.cpp Fathom/src/tbprobe.c
OBJS = main.o nnue_inference.o mcts.o tablebase.o tbprobe.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tbprobe.o: Fathom/src/tbprobe.c
	$(CXX) $(CXXFLAGS) -c Fathom/src/tbprobe.c -o tbprobe.o

clean:
	rm -f $(OBJS) $(TARGET)
