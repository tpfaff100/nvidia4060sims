# Auto-detects compiler by OS: macOS -> clang++, Linux -> g++.
# Override anytime:
#   make PLATFORM=macos   (force clang++)
#   make PLATFORM=linux   (force g++)
#   make CXX=g++          (force any specific compiler directly)
UNAME_S := $(shell uname -s)

ifeq ($(PLATFORM),macos)
  CXX := clang++
else ifeq ($(PLATFORM),linux)
  CXX := g++
else ifeq ($(UNAME_S),Darwin)
  CXX := clang++
else
  CXX := g++
endif

CXXFLAGS ?= -std=c++20 -O3
LDLIBS   ?= -lm

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:.cpp=)

.PHONY: all clean
all: $(BINS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS)
