# C Machine Learning Library From Scratch

Build a production-quality C library for learning machine learning from scratch.

## Project Overview

- **Name**: c-ml-from-scratch
- **Language**: Pure C (C11 standard)
- **Dependencies**: Zero external ML dependencies (only standard C library)
- **Purpose**: Educational ML library demonstrating core algorithms

## Directory Structure

```
c-ml-from-scratch/
├── include/           # Public headers
├── src/               # Implementation files
├── examples/          # Runnable CLI demos
├── tests/             # Unit tests
├── data/              # Sample CSV datasets
├── docs/              # Documentation
├── .github/workflows/ # CI configuration
├── CMakeLists.txt     # CMake build
├── Makefile           # Wrapper for convenience
├── README.md          # Project documentation
└── LICENSE            # MIT License
```

## Implementation Requirements

### Phase 1: Core Library (Matrix/Vector)

1. **Matrix Structure** (`include/matrix.h`, `src/matrix.c`)
   - `Matrix` struct with rows, cols, data pointer
   - `matrix_alloc()`, `matrix_free()`
   - `matrix_add()`, `matrix_sub()`, `matrix_mul()` (element-wise)
   - `matrix_matmul()` (matrix multiplication)
   - `matrix_transpose()`
   - `matrix_print()`

2. **Vector Operations** (`include/vector.h`, `src/vector.c`)
   - `vector_dot()` - dot product
   - `vector_norm()` - L2 norm
   - `vector_scale()` - scalar multiplication

3. **Utilities** (`include/utils.h`, `src/utils.c`)
   - `rand_uniform()`, `rand_normal()` - random number generation
   - `mean()`, `std()` - statistics

### Phase 2: Data Handling

4. **CSV Loader** (`include/csv.h`, `src/csv.c`)
   - `csv_load()` - load numeric CSV into Matrix
   - `csv_save()` - save Matrix to CSV

5. **Data Preprocessing** (`include/preprocessing.h`, `src/preprocessing.c`)
   - `train_test_split()` - split data into train/test sets
   - `standardize()` - z-score normalization
   - `minmax_scale()` - min-max scaling to [0,1]

### Phase 3: ML Algorithms

6. **Linear Regression** (`include/linear_regression.h`, `src/linear_regression.c`)
   - `linreg_fit_closed()` - closed-form solution (normal equation)
   - `linreg_fit_gd()` - gradient descent
   - `linreg_predict()`

7. **Logistic Regression** (`include/logistic_regression.h`, `src/logistic_regression.c`)
   - `logreg_fit()` - binary classification with gradient descent
   - `logreg_predict()` - probability output
   - `logreg_predict_class()` - class prediction (0/1)

8. **k-Nearest Neighbors** (`include/knn.h`, `src/knn.c`)
   - `knn_fit()` - store training data
   - `knn_predict()` - classification by majority vote

9. **k-Means Clustering** (`include/kmeans.h`, `src/kmeans.c`)
   - `kmeans_fit()` - Lloyd's algorithm
   - `kmeans_predict()` - assign cluster labels

### Phase 4: Evaluation Metrics

10. **Metrics** (`include/metrics.h`, `src/metrics.c`)
    - `mse()` - mean squared error
    - `accuracy()` - classification accuracy
    - `precision()`, `recall()`, `f1_score()` - binary classification metrics

### Phase 5: Examples

11. **Example Programs** (in `examples/`)
    - `linear_regression_example.c` - fit line to data
    - `logistic_regression_example.c` - binary classification
    - `knn_example.c` - classify with k-NN
    - `kmeans_example.c` - cluster data

### Phase 6: Testing

12. **Test Harness** (`tests/test_harness.h`)
    - Simple assert-based framework (no external dependencies)
    - `TEST()`, `ASSERT_EQ()`, `ASSERT_NEAR()` macros

13. **Unit Tests** (`tests/`)
    - `test_matrix.c` - matrix operation tests
    - `test_linreg.c` - linear regression sanity test with golden values

### Phase 7: Build System

14. **CMakeLists.txt**
    - Build static library `libcml.a`
    - Build examples
    - Build and run tests with `ctest`

15. **Makefile** (wrapper)
    - `make` - build all
    - `make test` - run tests
    - `make examples` - build examples
    - `make clean` - clean build

### Phase 8: Documentation & CI

16. **README.md**
    - Project description
    - Build instructions
    - Usage examples
    - API reference

17. **docs/ROADMAP.md**
    - Future enhancements
    - Potential additions (neural networks, decision trees, etc.)

18. **GitHub Actions** (`.github/workflows/ci.yml`)
    - Build on Ubuntu
    - Run tests
    - Report status

19. **.gitignore** - ignore build artifacts
20. **LICENSE** - MIT license

## Sample Data

Create in `data/`:
- `simple_linear.csv` - 2D data for linear regression
- `binary_classification.csv` - labeled data for logistic regression/kNN
- `clusters.csv` - unlabeled data for k-means

## Completion Criteria

The project is COMPLETE when:
- [ ] All source files implemented
- [ ] CMake builds without errors
- [ ] All tests pass via ctest
- [ ] Examples run successfully
- [ ] README documents usage
- [ ] GitHub Actions CI passes

When ALL above criteria are satisfied, output:
<promise>COMPLETE</promise>
