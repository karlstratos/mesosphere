# Author: Karl Stratos (me@karlstratos.com)


############ Configuration #####################################################
CORE = ..
BUILD = build
GTEST = ../../third_party/googletest
EIGEN = ../../third_party/eigen-eigen-5a0156e40feb

CC = clang++
WARN = -Wall
OPT =
CFLAGS = $(WARN) $(OPT) -std=c++11
LIB =
ifeq ($(shell uname), Linux)
	LIB += -lpthread
endif
INC = -I $(EIGEN) -isystem $(GTEST)/include

TARGETS = graph_test util_eigen_test util_test
################################################################################


all: $(TARGETS)

dir:
	@mkdir -p $(BUILD)

.PHONY: clean
clean:
	rm -rf $(BUILD) $(TARGETS) *.dSYM

# Linking
graph_test: dir $(BUILD)/graph_test.o $(BUILD)/graph.o $(BUILD)/gtest_main.a
	$(CC) $(filter-out dir,$^) -o $@ $(LIB)

util_eigen_test: dir $(BUILD)/util_eigen_test.o $(BUILD)/gtest_main.a
	$(CC) $(filter-out dir,$^) -o $@ $(LIB)

util_test: dir $(BUILD)/util_test.o $(BUILD)/gtest_main.a
	$(CC) $(filter-out dir,$^) -o $@ $(LIB)

# Compiling
$(BUILD)/graph_test.o: graph_test.cc
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/graph.o: $(CORE)/graph.* $(CORE)/util.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/util_eigen_test.o: util_eigen_test.cc $(CORE)/util_eigen.h \
	$(CORE)/util.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/util_test.o: util_test.cc $(CORE)/util.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@


########### Build gtest_main.a: no need to touch this. #########################
GTEST_HEADERS = $(GTEST)/include/gtest/*.h $(GTEST)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST)/src/*.cc $(GTEST)/src/*.h $(GTEST_HEADERS)

$(BUILD)/gtest_main.a: $(BUILD)/gtest-all.o $(BUILD)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(BUILD)/gtest-all.o: $(GTEST_SRCS_)
	$(CC) -I $(GTEST) $(CFLAGS) $(INC) -c $(GTEST)/src/gtest-all.cc -o $@

$(BUILD)/gtest_main.o: $(GTEST_SRCS_)
	$(CC) -I $(GTEST) $(CFLAGS) $(INC) -c $(GTEST)/src/gtest_main.cc -o $@
################################################################################
