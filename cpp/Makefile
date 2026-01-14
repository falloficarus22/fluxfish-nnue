CXX = g++
CXXFLAGS = -O3 -mavx2 -mfma -Wall -std=c++17

TARGET = fluxfish_cpp
SRCS = main.cpp nnue_inference.cpp mcts.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
