# Author: Karl Stratos (me@karlstratos.com)


############ Configuration #####################################################
SRC = ..
BUILD = build
GTEST = ../../third_party/googletest

CC = clang++
WARN = -Wall
OPT =
CFLAGS = $(WARN) $(OPT) -std=c++11
LIB =
ifeq ($(shell uname), Linux)
	LIB += -lpthread
endif
INC = -isystem $(GTEST)/include

TARGETS = util_test
################################################################################


all: $(TARGETS)

dir:
	@mkdir -p $(BUILD)

.PHONY: clean
clean:
	rm -rf $(BUILD) $(TARGETS) *.dSYM

# Linking
util_test: dir $(BUILD)/util_test.o $(BUILD)/gtest_main.a
	$(CC) $(filter-out dir,$^) -o $@ $(LIB)

# Compiling
$(BUILD)/util_test.o: util_test.cc
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
