# Author: Karl Stratos (me@karlstratos.com)

############ Configuration #####################################################
CORE = ../../core
BUILD = build
EIGEN = ../../third_party/eigen-eigen-5a0156e40feb

CC = clang++
WARN = -Wall
OPT = -Ofast
CFLAGS = $(WARN) $(OPT) -std=c++11
LIB =
ifeq ($(shell uname), Linux)
	LIB += -lpthread
endif
INC = -I $(EIGEN) -isystem $(GTEST)/include

TARGETS = main
################################################################################


all: $(TARGETS)

dir:
	@mkdir -p $(BUILD)

.PHONY: clean
clean:
	rm -rf $(BUILD) $(TARGETS) *.dSYM


# Linking
main: dir $(BUILD)/main.o $(BUILD)/autodiff.o $(BUILD)/dag.o
	$(CC) $(filter-out dir,$^) -o $@ $(LIB)

# Compiling
$(BUILD)/main.o: main.cc
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/autodiff.o: $(CORE)/autodiff.* $(CORE)/dag.* $(CORE)/util_eigen.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/dag.o: $(CORE)/dag.* $(CORE)/util.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
