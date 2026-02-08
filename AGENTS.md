# Agent Operations Guide

Quick reference for building and testing this project.

## Build Commands

```bash
# Configure and build (from project root)
mkdir -p build && cd build
cmake ..
make

# Or use Makefile wrapper
make          # Build everything
make test     # Run tests
make examples # Build examples only
make clean    # Clean build artifacts
```

## Test Commands

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run specific test
./build/tests/test_matrix
./build/tests/test_linreg
```

## Run Examples

```bash
./build/examples/linear_regression_example
./build/examples/logistic_regression_example
./build/examples/knn_example
./build/examples/kmeans_example
```

## Project Structure

```
include/    - Public headers (*.h)
src/        - Implementation (*.c)
examples/   - Demo programs
tests/      - Unit tests
data/       - Sample CSV files
docs/       - Documentation
```

## Key Files to Implement

1. `include/matrix.h` + `src/matrix.c` - Core matrix operations
2. `include/csv.h` + `src/csv.c` - Data loading
3. `include/linear_regression.h` + `src/linear_regression.c` - Linear regression
4. `include/logistic_regression.h` + `src/logistic_regression.c` - Logistic regression
5. `include/knn.h` + `src/knn.c` - k-Nearest Neighbors
6. `include/kmeans.h` + `src/kmeans.c` - k-Means clustering
7. `include/metrics.h` + `src/metrics.c` - Evaluation metrics

## Coding Standards

- C11 standard
- No external dependencies (stdlib only)
- Use `const` where appropriate
- Check malloc returns for NULL
- Free all allocated memory
- Document public functions with comments

## Iteration Strategy

1. Start with matrix operations (foundation)
2. Add CSV loader and preprocessing
3. Implement algorithms one by one
4. Add tests for each component
5. Create examples
6. Write documentation
7. Set up CI

## Verification Checklist

Before claiming completion:
- [ ] `make` succeeds without warnings
- [ ] `make test` passes all tests
- [ ] Examples produce expected output
- [ ] README has build/usage instructions
- [ ] CI workflow file exists and is valid
