# tinycml Makefile
# Tiny C Machine Learning Library - Direct compilation without CMake dependency

CC = cc
CFLAGS = -std=c11 -Wall -Wextra -pedantic -O2
LDFLAGS = -lm

BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
TEST_DIR = $(BUILD_DIR)/tests
EXAMPLES_DIR = $(BUILD_DIR)/examples

INCLUDE = -Iinclude

# Source files
LIB_SRCS = $(wildcard src/*.c)
LIB_OBJS = $(patsubst src/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))

# Library
LIBRARY = $(LIB_DIR)/libtinycml.a

# Examples
EXAMPLES = linear_regression_example logistic_regression_example knn_example kmeans_example estimator_api_example cross_validation_example pipeline_example random_forest_example pca_example feature_selection_example
EXAMPLE_BINS = $(patsubst %,$(EXAMPLES_DIR)/%,$(EXAMPLES))

# Tests
TESTS = test_matrix test_linreg
TEST_BINS = $(patsubst %,$(TEST_DIR)/%,$(TESTS))

.PHONY: all build library examples tests test clean

all: build

build: library examples tests

# Create directories
$(BUILD_DIR) $(LIB_DIR) $(OBJ_DIR) $(BIN_DIR) $(TEST_DIR) $(EXAMPLES_DIR):
	mkdir -p $@

# Compile library objects
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Create static library
library: $(LIBRARY)

$(LIBRARY): $(LIB_OBJS) | $(LIB_DIR)
	ar rcs $@ $^

# Build examples
examples: $(EXAMPLE_BINS)

$(EXAMPLES_DIR)/%: examples/%.c $(LIBRARY) | $(EXAMPLES_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) $< -L$(LIB_DIR) -ltinycml $(LDFLAGS) -o $@

# Build tests
tests: $(TEST_BINS)

$(TEST_DIR)/%: tests/%.c $(LIBRARY) | $(TEST_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -Itests $< -L$(LIB_DIR) -ltinycml $(LDFLAGS) -o $@

# Run tests
test: tests
	@echo "Running tests..."
	@for test in $(TEST_BINS); do \
		echo ""; \
		echo "=== Running $$test ==="; \
		$$test || exit 1; \
	done
	@echo ""
	@echo "All tests passed!"

clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build everything (library, examples, tests)"
	@echo "  build    - Same as all"
	@echo "  library  - Build static library only"
	@echo "  examples - Build example programs"
	@echo "  tests    - Build test programs"
	@echo "  test     - Build and run tests"
	@echo "  clean    - Remove build directory"
