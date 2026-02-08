# API Reference

Complete API documentation for tinycml.

## Table of Contents

1. [Matrix Operations](#matrix-operations)
2. [Vector Operations](#vector-operations)
3. [Utility Functions](#utility-functions)
4. [CSV Handling](#csv-handling)
5. [Data Preprocessing](#data-preprocessing)
6. [Linear Regression](#linear-regression)
7. [Logistic Regression](#logistic-regression)
8. [k-Nearest Neighbors](#k-nearest-neighbors)
9. [k-Means Clustering](#k-means-clustering)
10. [Evaluation Metrics](#evaluation-metrics)

---

## Matrix Operations

**Header:** `matrix.h`

### Data Structure

```c
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of columns
    double *data;   // Row-major data array
} Matrix;
```

**Memory Layout:** Row-major order. Element (i, j) is at `data[i * cols + j]`.

### Memory Management

#### `matrix_alloc`
```c
Matrix* matrix_alloc(size_t rows, size_t cols);
```
Allocate a zero-initialized matrix.

**Parameters:**
- `rows` - Number of rows
- `cols` - Number of columns

**Returns:** Pointer to allocated matrix, or `NULL` on failure.

**Example:**
```c
Matrix *m = matrix_alloc(3, 4);  // 3x4 matrix of zeros
if (m == NULL) {
    fprintf(stderr, "Allocation failed\n");
}
```

#### `matrix_free`
```c
void matrix_free(Matrix *m);
```
Free matrix memory. Safe to call with `NULL`.

#### `matrix_copy`
```c
Matrix* matrix_copy(const Matrix *m);
```
Create a deep copy of a matrix.

### Element Access

#### `matrix_get`
```c
double matrix_get(const Matrix *m, size_t i, size_t j);
```
Get element at position (i, j). Asserts bounds in debug mode.

#### `matrix_set`
```c
void matrix_set(Matrix *m, size_t i, size_t j, double val);
```
Set element at position (i, j).

### Arithmetic Operations

#### `matrix_add`
```c
Matrix* matrix_add(const Matrix *a, const Matrix *b);
```
Element-wise addition. Returns `NULL` on dimension mismatch.

#### `matrix_sub`
```c
Matrix* matrix_sub(const Matrix *a, const Matrix *b);
```
Element-wise subtraction.

#### `matrix_mul`
```c
Matrix* matrix_mul(const Matrix *a, const Matrix *b);
```
Element-wise multiplication (Hadamard product).

#### `matrix_scale`
```c
Matrix* matrix_scale(const Matrix *m, double scalar);
```
Multiply all elements by scalar.

#### `matrix_matmul`
```c
Matrix* matrix_matmul(const Matrix *a, const Matrix *b);
```
Matrix multiplication. For A(m×k) and B(k×n), returns C(m×n).

**Complexity:** O(m × n × k)

### Transformations

#### `matrix_transpose`
```c
Matrix* matrix_transpose(const Matrix *m);
```
Return transposed matrix.

### Utilities

#### `matrix_print`
```c
void matrix_print(const Matrix *m);
```
Print matrix to stdout in formatted form.

#### `matrix_fill`
```c
void matrix_fill(Matrix *m, double val);
```
Fill all elements with a constant value.

#### `matrix_identity`
```c
Matrix* matrix_identity(size_t n);
```
Create n×n identity matrix.

---

## Vector Operations

**Header:** `vector.h`

Vectors are represented as (n×1) or (1×n) matrices.

### Functions

#### `vector_dot`
```c
double vector_dot(const Matrix *a, const Matrix *b);
```
Compute dot product. Works with any matrix shape (flattens both).

#### `vector_norm`
```c
double vector_norm(const Matrix *v);
```
Compute L2 (Euclidean) norm: `||v|| = sqrt(sum(v_i^2))`.

#### `vector_scale`
```c
Matrix* vector_scale(const Matrix *v, double scalar);
```
Scale vector by scalar (alias for `matrix_scale`).

#### `vector_add` / `vector_sub`
```c
Matrix* vector_add(const Matrix *a, const Matrix *b);
Matrix* vector_sub(const Matrix *a, const Matrix *b);
```
Element-wise vector operations.

---

## Utility Functions

**Header:** `utils.h`

### Random Number Generation

#### `rand_seed`
```c
void rand_seed(unsigned int seed);
```
Seed the random number generator for reproducibility.

#### `rand_uniform`
```c
double rand_uniform(void);
```
Generate uniform random number in [0, 1).

#### `rand_uniform_range`
```c
double rand_uniform_range(double min, double max);
```
Generate uniform random number in [min, max).

#### `rand_normal`
```c
double rand_normal(void);
```
Generate standard normal random number (mean=0, std=1) using Box-Muller transform.

#### `rand_normal_params`
```c
double rand_normal_params(double mean, double std);
```
Generate normal random number with specified mean and standard deviation.

### Statistics

#### `mean`
```c
double mean(const double *data, size_t n);
```
Compute arithmetic mean.

#### `std_dev`
```c
double std_dev(const double *data, size_t n);
```
Compute sample standard deviation (with Bessel's correction).

#### `variance`
```c
double variance(const double *data, size_t n);
```
Compute sample variance.

#### `shuffle_indices`
```c
void shuffle_indices(size_t *indices, size_t n);
```
Fisher-Yates shuffle for an array of indices.

---

## CSV Handling

**Header:** `csv.h`

### Functions

#### `csv_load`
```c
Matrix* csv_load(const char *filename, int has_header);
```
Load CSV file into a matrix.

**Parameters:**
- `filename` - Path to CSV file
- `has_header` - 1 if first row is header (will be skipped), 0 otherwise

**Returns:** Matrix with data, or `NULL` on error.

**Supported formats:**
- Comma-separated values
- Numeric data only (doubles)
- Unix or Windows line endings

**Example:**
```c
// Load CSV with header
Matrix *data = csv_load("data/iris.csv", 1);

// Load CSV without header
Matrix *data = csv_load("data/numbers.csv", 0);
```

#### `csv_save`
```c
int csv_save(const Matrix *m, const char *filename);
```
Save matrix to CSV file.

**Returns:** 0 on success, -1 on error.

---

## Data Preprocessing

**Header:** `preprocessing.h`

### Train/Test Split

```c
typedef struct {
    Matrix *X_train;
    Matrix *X_test;
    Matrix *y_train;
    Matrix *y_test;
} TrainTestSplit;

TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed);
void train_test_split_free(TrainTestSplit *split);
```

**Parameters:**
- `X` - Feature matrix (n_samples × n_features)
- `y` - Target vector (n_samples × 1)
- `test_ratio` - Fraction for test set (e.g., 0.2 for 20%)
- `seed` - Random seed for reproducibility

**Example:**
```c
TrainTestSplit split = train_test_split(X, y, 0.2, 42);
// Use split.X_train, split.y_train for training
// Use split.X_test, split.y_test for testing
train_test_split_free(&split);
```

### Standardization (Z-Score)

```c
typedef struct {
    double *means;
    double *stds;
    size_t n_features;
} Scaler;

Scaler* standardize_fit(const Matrix *X);
Matrix* standardize_transform(const Matrix *X, const Scaler *scaler);
Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out);
void scaler_free(Scaler *scaler);
```

**Formula:** `z = (x - mean) / std`

**Example:**
```c
Scaler *scaler = NULL;
Matrix *X_scaled = standardize_fit_transform(X_train, &scaler);
Matrix *X_test_scaled = standardize_transform(X_test, scaler);
scaler_free(scaler);
```

### Min-Max Scaling

```c
typedef struct {
    double *mins;
    double *maxs;
    size_t n_features;
} MinMaxScaler;

MinMaxScaler* minmax_fit(const Matrix *X);
Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler);
void minmax_scaler_free(MinMaxScaler *scaler);
```

**Formula:** `x_scaled = (x - min) / (max - min)`

Scales features to [0, 1] range.

### Bias Column

```c
Matrix* add_bias_column(const Matrix *X);
```

Add a column of ones as the first column (for regression intercept).

**Example:**
```c
// X is (n × m), result is (n × m+1) with first column = 1.0
Matrix *X_bias = add_bias_column(X);
```

---

## Linear Regression

**Header:** `linear_regression.h`

### Functions

#### `linreg_fit_closed`
```c
Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y);
```
Fit linear regression using closed-form solution (normal equation).

**Formula:** `w = (X'X)^(-1) X'y`

**Parameters:**
- `X` - Feature matrix with bias column (n × m)
- `y` - Target vector (n × 1)

**Returns:** Weight vector (m × 1)

**Note:** X should include bias column. Use `add_bias_column()` first.

#### `linreg_fit_gd`
```c
Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs);
```
Fit linear regression using gradient descent.

**Update rule:** `w = w - lr * X' * (X*w - y) / n`

**Parameters:**
- `X` - Feature matrix with bias column
- `y` - Target vector
- `lr` - Learning rate (e.g., 0.01)
- `epochs` - Number of iterations

#### `linreg_predict`
```c
Matrix* linreg_predict(const Matrix *X, const Matrix *weights);
```
Predict target values: `y_pred = X * weights`

---

## Logistic Regression

**Header:** `logistic_regression.h`

### Functions

#### `sigmoid`
```c
double sigmoid(double x);
```
Compute sigmoid function: `1 / (1 + exp(-x))`

Includes overflow protection for numerical stability.

#### `logreg_fit`
```c
Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs);
```
Fit logistic regression for binary classification.

**Parameters:**
- `X` - Feature matrix with bias column
- `y` - Binary target vector (values 0 or 1)
- `lr` - Learning rate
- `epochs` - Number of iterations

**Gradient:** `gradient = X' * (sigmoid(X*w) - y) / n`

#### `logreg_predict_proba`
```c
Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights);
```
Predict probabilities P(y=1|x).

#### `logreg_predict`
```c
Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold);
```
Predict class labels (0 or 1) using threshold (typically 0.5).

---

## k-Nearest Neighbors

**Header:** `knn.h`

### Data Structure

```c
typedef struct {
    Matrix *X_train;  // Training features
    Matrix *y_train;  // Training labels
    int k;            // Number of neighbors
} KNNModel;
```

### Functions

#### `knn_fit`
```c
KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k);
```
Create k-NN model (stores training data).

**Parameters:**
- `X` - Training features
- `y` - Training labels (class indices as doubles)
- `k` - Number of neighbors

#### `knn_predict`
```c
Matrix* knn_predict(const KNNModel *model, const Matrix *X);
```
Predict class labels using majority vote.

**Algorithm:**
1. For each test sample, compute Euclidean distance to all training samples
2. Find k nearest neighbors
3. Return majority class among neighbors

#### `knn_free`
```c
void knn_free(KNNModel *model);
```
Free k-NN model memory.

---

## k-Means Clustering

**Header:** `kmeans.h`

### Data Structure

```c
typedef struct {
    Matrix *centroids;  // Cluster centroids (k × n_features)
    int k;              // Number of clusters
    int max_iter;       // Maximum iterations
} KMeansModel;
```

### Functions

#### `kmeans_fit`
```c
KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed);
```
Fit k-Means model using Lloyd's algorithm.

**Parameters:**
- `X` - Data matrix (n_samples × n_features)
- `k` - Number of clusters
- `max_iter` - Maximum iterations
- `seed` - Random seed for centroid initialization

**Algorithm:**
1. Initialize k centroids randomly from data points
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence or max_iter

#### `kmeans_predict`
```c
Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X);
```
Assign cluster labels to samples.

#### `kmeans_free`
```c
void kmeans_free(KMeansModel *model);
```
Free k-Means model memory.

---

## Evaluation Metrics

**Header:** `metrics.h`

### Regression Metrics

#### `mse`
```c
double mse(const Matrix *y_true, const Matrix *y_pred);
```
Mean Squared Error: `(1/n) * sum((y_true - y_pred)^2)`

#### `rmse`
```c
double rmse(const Matrix *y_true, const Matrix *y_pred);
```
Root Mean Squared Error: `sqrt(MSE)`

#### `mae`
```c
double mae(const Matrix *y_true, const Matrix *y_pred);
```
Mean Absolute Error: `(1/n) * sum(|y_true - y_pred|)`

### Classification Metrics

#### `accuracy`
```c
double accuracy(const Matrix *y_true, const Matrix *y_pred);
```
Classification accuracy: `correct / total`

#### `precision`
```c
double precision(const Matrix *y_true, const Matrix *y_pred);
```
Precision for binary classification: `TP / (TP + FP)`

#### `recall`
```c
double recall(const Matrix *y_true, const Matrix *y_pred);
```
Recall for binary classification: `TP / (TP + FN)`

#### `f1_score`
```c
double f1_score(const Matrix *y_true, const Matrix *y_pred);
```
F1 Score: `2 * (precision * recall) / (precision + recall)`

### Confusion Matrix

```c
typedef struct {
    int tp;  // True Positives
    int tn;  // True Negatives
    int fp;  // False Positives
    int fn;  // False Negatives
} ConfusionMatrix;

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);
void confusion_matrix_print(const ConfusionMatrix *cm);
```

**Example:**
```c
ConfusionMatrix cm = confusion_matrix(y_test, predictions);
printf("Accuracy: %.2f%%\n", (double)(cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn) * 100);
confusion_matrix_print(&cm);
```
