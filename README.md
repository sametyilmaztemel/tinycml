<p align="center">
  <img src="assets/logo.png" alt="tinycml logo" width="300">
</p>

<h1 align="center">tinycml</h1>

<p align="center">
  <strong>Tiny C Machine Learning Library</strong>
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

# English

A comprehensive, zero-dependency C library implementing scikit-learn style machine learning in pure C.

## Overview

tinycml implements a wide range of machine learning algorithms in pure C (C11 standard) with zero external dependencies. It provides a unified scikit-learn style API (`fit`/`predict`/`score`) while maintaining C's advantages: instant startup, tiny binary size, and embedded system compatibility.

## Library Statistics

| Metric | Value |
|--------|-------|
| **Library Size** | ~160KB |
| **Lines of Code** | ~9,700 |
| **Dependencies** | Zero |
| **Startup Time** | ~1ms |
| **C Standard** | C11 |

## Why tinycml? Advantages Over Modern ML Libraries

### 🎯 Educational Value

| Aspect | tinycml | TensorFlow/PyTorch/scikit-learn |
|--------|---------|--------------------------------|
| **Code Transparency** | Every algorithm is readable, ~100-300 lines each | Thousands of lines, heavy abstractions |
| **Dependencies** | Zero (only standard C library) | Hundreds of packages, complex environments |
| **Understanding** | See exactly how gradient descent, backprop work | Black-box functions |
| **Debugging** | Step through with any C debugger | Complex stack traces |

### 🚀 Performance Characteristics

| Feature | tinycml | Python ML Libraries |
|---------|---------|---------------------|
| **Startup Time** | Instant (~1ms) | Seconds (import overhead) |
| **Memory Footprint** | ~160KB binary | 100MB+ with dependencies |
| **No GIL** | True parallelism possible | Python GIL limitations |
| **Embedded Systems** | Runs on microcontrollers | Requires full OS |

### 🔧 Use Cases Where This Library Excels

1. **Learning ML Fundamentals**: Understand the math behind algorithms
2. **Embedded/IoT Devices**: Run ML on resource-constrained hardware
3. **Real-time Systems**: Predictable, low-latency inference
4. **Custom Modifications**: Easy to extend and modify algorithms
5. **No-dependency Environments**: Air-gapped systems, minimal containers

## Features

### Core Infrastructure
- **Unified Estimator API**: scikit-learn style `fit`/`predict`/`score` interface
- **Pipeline System**: Chain preprocessing steps with models
- **Cross-Validation**: K-Fold, Stratified K-Fold with scoring
- **Model Selection**: GridSearchCV for hyperparameter tuning
- **Model Serialization**: Save/load trained models to binary files

### Supervised Learning
- **Linear Regression** (closed-form and gradient descent)
- **Logistic Regression** (binary classification with L2 regularization)
- **k-Nearest Neighbors** (classification and regression)
- **Naive Bayes** (Gaussian)
- **Decision Tree** (classification with Gini/Entropy criteria)
- **Random Forest** (ensemble with bootstrap, OOB score)
- **Neural Network** (feedforward with backpropagation, multiple activations)
- **Support Vector Machine** (linear SVM)

### Unsupervised Learning
- **k-Means Clustering** (with k-means++ initialization)
- **PCA** (Principal Component Analysis with whitening)

### Feature Engineering
- **Feature Selection**: SelectKBest, VarianceThreshold
- **Scoring Functions**: f_classif, f_regression, chi2, mutual_info
- **Preprocessing**: StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures

### Evaluation
- **Regression Metrics**: MSE, RMSE, MAE, R²
- **Classification Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Clustering Metrics**: Inertia, Silhouette Score

## Building

### Prerequisites

- C11 compatible compiler (GCC, Clang, MSVC)
- Make (optional, for convenience)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sametyilmaztemel/tinycml.git
cd tinycml

# Build everything
make

# Run tests
make test

# Run examples
./build/examples/linear_regression_example
./build/examples/random_forest_example
./build/examples/neural_network_example
./build/examples/pca_example
./build/examples/feature_selection_example
```

### Build Options

```bash
make build     # Build library, examples, and tests
make library   # Build only the static library
make examples  # Build example programs
make tests     # Build test programs
make test      # Build and run all tests
make clean     # Remove build artifacts
```

## Usage Guide

### The Unified Estimator API

All models in tinycml follow a consistent interface inspired by scikit-learn:

```c
#include "estimator.h"
#include "linear_regression.h"

// Create model
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Train
model->base.fit((Estimator*)model, X_train, y_train);

// Predict
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Evaluate
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Free
model->base.free((Estimator*)model);
```

### Pipeline: Chain Preprocessing with Models

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "linear_regression.h"

// Create pipeline with preprocessing + model
Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "model", (Estimator*)linear_regression_create(LINREG_SOLVER_CLOSED));

// Fit entire pipeline
pipe->base.fit((Estimator*)pipe, X_train, y_train);

// Predict (automatically applies all transformations)
Matrix *pred = pipe->base.predict((Estimator*)pipe, X_test);

// Score
double score = pipe->base.score((Estimator*)pipe, X_test, y_test);

pipeline_free(pipe);
```

### Cross-Validation

```c
#include "validation.h"
#include "logistic_regression.h"

LogisticRegression *model = logistic_regression_create_full(0.01, 1000, 0.0);

// 5-fold cross-validation
CrossValResults *cv = cross_val_score((Estimator*)model, X, y, 5, 1, 42);

printf("Mean accuracy: %.4f (+/- %.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
model->base.free((Estimator*)model);
```

### Hyperparameter Tuning with GridSearchCV

```c
#include "model_selection.h"
#include "decision_tree.h"

// Define parameter grid
ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){3, 5, 10}, 3);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

// Create GridSearchCV
DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

// Fit (searches all parameter combinations)
gs->base.fit((Estimator*)gs, X, y);

printf("Best score: %.4f\n", gs->best_score_);
printf("Best max_depth: %d\n", grid_search_get_best_int(gs, "max_depth"));

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### Random Forest

```c
#include "ensemble.h"

// Create Random Forest with 100 trees
RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,    // n_estimators
    10,     // max_depth
    2,      // min_samples_split
    1,      // min_samples_leaf
    0,      // max_features (0 = sqrt)
    1,      // bootstrap
    42      // random_state
);

rf->base.fit((Estimator*)rf, X_train, y_train);

double accuracy = rf->base.score((Estimator*)rf, X_test, y_test);
printf("Test accuracy: %.4f\n", accuracy);
printf("OOB score: %.4f\n", rf->oob_score_);

// Probability predictions
Matrix *proba = rf->base.predict_proba((Estimator*)rf, X_test);

rf->base.free((Estimator*)rf);
```

### Neural Network

```c
#include "neural_network.h"

// Create network: input -> 64 -> 32 -> output
size_t layer_sizes[] = {n_features, 64, 32, n_classes};
NeuralNetwork *nn = neural_network_create(layer_sizes, 4, ACTIVATION_RELU);

// Configure training
nn->learning_rate = 0.001;
nn->epochs = 100;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);

double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);
printf("Neural network accuracy: %.4f\n", accuracy);

nn->base.free((Estimator*)nn);
```

### PCA (Dimensionality Reduction)

```c
#include "decomposition.h"

// Reduce to 2 principal components
PCA *pca = pca_create(2);
pca->base.fit((Estimator*)pca, X, NULL);

// Transform data
Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);

// Check explained variance
const double *evr = pca_explained_variance_ratio(pca);
printf("PC1 explains %.2f%% of variance\n", evr[0] * 100);
printf("PC2 explains %.2f%% of variance\n", evr[1] * 100);

// Reconstruct original data
Matrix *X_reconstructed = pca_inverse_transform(pca, X_reduced);

pca->base.free((Estimator*)pca);
```

### Feature Selection

```c
#include "feature_selection.h"

// SelectKBest: Keep top 5 features by F-score
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 5);
selector->base.fit((Estimator*)selector, X, y);

// Get selected feature indices
const int *support = select_k_best_get_support(selector);

// Transform data to selected features only
Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// VarianceThreshold: Remove low-variance features
VarianceThreshold *vt = variance_threshold_create(0.1);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

### Model Serialization

```c
// Save trained model
model->base.save((Estimator*)model, "model.bin");

// Load model
LinearRegression *loaded = (LinearRegression*)linear_regression_load("model.bin");
```

### Training Progress and Callbacks

```c
#include "estimator.h"

// Enable verbose output
model->base.verbose = VERBOSE_PROGRESS;

// Or use custom callback
void my_callback(int epoch, double loss, double metric, void *data) {
    printf("Epoch %d: loss=%.4f, metric=%.4f\n", epoch, loss, metric);
}

estimator_set_callback((Estimator*)model, my_callback, NULL);

// After training, access history
const TrainingHistory *history = estimator_get_history((Estimator*)model);
```

## Examples

The library includes comprehensive examples:

| Example | Description |
|---------|-------------|
| `linear_regression_example` | Closed-form vs gradient descent |
| `logistic_regression_example` | Binary classification |
| `knn_example` | k-Nearest Neighbors |
| `kmeans_example` | Clustering with k-means++ |
| `estimator_api_example` | Unified API demonstration |
| `cross_validation_example` | K-Fold cross-validation |
| `pipeline_example` | Preprocessing + model chains |
| `random_forest_example` | Ensemble learning |
| `pca_example` | Dimensionality reduction |
| `feature_selection_example` | Feature importance and selection |

## Project Structure

```
tinycml/
├── include/              # Public headers
│   ├── matrix.h          # Matrix operations
│   ├── estimator.h       # Unified estimator API
│   ├── pipeline.h        # Pipeline system
│   ├── validation.h      # Cross-validation
│   ├── model_selection.h # GridSearchCV
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── ensemble.h        # Random Forest
│   ├── neural_network.h
│   ├── decomposition.h   # PCA
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/                  # Implementation files
├── examples/             # Runnable demos
├── tests/                # Unit tests
├── data/                 # Sample datasets
└── docs/                 # Documentation
```

## License

MIT License - see LICENSE file for details.

---

# Türkçe

Saf C ile scikit-learn tarzı makine öğrenmesi uygulayan kapsamlı, sıfır bağımlılıklı bir C kütüphanesi.

## Genel Bakış

tinycml, geniş bir makine öğrenmesi algoritması yelpazesini saf C (C11 standardı) ile sıfır harici bağımlılık kullanarak uygular. Birleşik scikit-learn tarzı API (`fit`/`predict`/`score`) sunarken C'nin avantajlarını korur: anlık başlangıç, küçük binary boyutu ve gömülü sistem uyumluluğu.

## Kütüphane İstatistikleri

| Metrik | Değer |
|--------|-------|
| **Kütüphane Boyutu** | ~160KB |
| **Kod Satırı** | ~9,700 |
| **Bağımlılık** | Sıfır |
| **Başlangıç Süresi** | ~1ms |
| **C Standardı** | C11 |

## Özellikler

### Temel Altyapı
- **Birleşik Estimator API'si**: scikit-learn tarzı `fit`/`predict`/`score` arayüzü
- **Pipeline Sistemi**: Ön işleme adımlarını modellerle zincirleyin
- **Çapraz Doğrulama**: K-Fold, Stratified K-Fold
- **Model Seçimi**: Hiperparametre ayarı için GridSearchCV
- **Model Serileştirme**: Eğitilmiş modelleri kaydet/yükle

### Denetimli Öğrenme
- **Lineer Regresyon** (kapalı form ve gradient descent)
- **Lojistik Regresyon** (L2 düzenlileştirmeli ikili sınıflandırma)
- **k-En Yakın Komşu** (sınıflandırma ve regresyon)
- **Naive Bayes** (Gaussian)
- **Karar Ağacı** (Gini/Entropi kriterleriyle sınıflandırma)
- **Rastgele Orman** (bootstrap ile topluluk, OOB skoru)
- **Sinir Ağı** (geri yayılım ile ileri beslemeli, çoklu aktivasyonlar)
- **Destek Vektör Makinesi** (lineer SVM)

### Denetimsiz Öğrenme
- **k-Means Kümeleme** (k-means++ başlatma ile)
- **PCA** (Beyazlatma ile Temel Bileşen Analizi)

### Özellik Mühendisliği
- **Özellik Seçimi**: SelectKBest, VarianceThreshold
- **Puanlama Fonksiyonları**: f_classif, f_regression, chi2, mutual_info
- **Ön İşleme**: StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures

### Değerlendirme
- **Regresyon Metrikleri**: MSE, RMSE, MAE, R²
- **Sınıflandırma Metrikleri**: Doğruluk, Kesinlik, Duyarlılık, F1, Karışıklık Matrisi
- **Kümeleme Metrikleri**: Atalet, Silhouette Skoru

## Derleme

### Gereksinimler

- C11 uyumlu derleyici (GCC, Clang, MSVC)
- Make (isteğe bağlı)

### Hızlı Başlangıç

```bash
# Depoyu klonlayın
git clone https://github.com/sametyilmaztemel/tinycml.git
cd tinycml

# Her şeyi derleyin
make

# Testleri çalıştırın
make test

# Örnekleri çalıştırın
./build/examples/linear_regression_example
./build/examples/random_forest_example
./build/examples/neural_network_example
./build/examples/pca_example
./build/examples/feature_selection_example
```

## Kullanım Rehberi

### Birleşik Estimator API'si

tinycml'deki tüm modeller scikit-learn'den esinlenen tutarlı bir arayüz izler:

```c
#include "estimator.h"
#include "linear_regression.h"

// Model oluştur
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Eğit
model->base.fit((Estimator*)model, X_train, y_train);

// Tahmin et
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Değerlendir
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Serbest bırak
model->base.free((Estimator*)model);
```

### Pipeline: Ön İşlemeyi Modellerle Zincirleyin

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "linear_regression.h"

// Ön işleme + model ile pipeline oluştur
Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "model", (Estimator*)linear_regression_create(LINREG_SOLVER_CLOSED));

// Tüm pipeline'ı eğit
pipe->base.fit((Estimator*)pipe, X_train, y_train);

// Tahmin et (tüm dönüşümleri otomatik uygular)
Matrix *pred = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

### Çapraz Doğrulama

```c
#include "validation.h"
#include "logistic_regression.h"

LogisticRegression *model = logistic_regression_create_full(0.01, 1000, 0.0);

// 5-katlı çapraz doğrulama
CrossValResults *cv = cross_val_score((Estimator*)model, X, y, 5, 1, 42);

printf("Ortalama doğruluk: %.4f (+/- %.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
model->base.free((Estimator*)model);
```

### GridSearchCV ile Hiperparametre Ayarı

```c
#include "model_selection.h"
#include "decision_tree.h"

// Parametre ızgarası tanımla
ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){3, 5, 10}, 3);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

// GridSearchCV oluştur
DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

// Eğit (tüm parametre kombinasyonlarını arar)
gs->base.fit((Estimator*)gs, X, y);

printf("En iyi skor: %.4f\n", gs->best_score_);

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### Rastgele Orman

```c
#include "ensemble.h"

// 100 ağaçlı Rastgele Orman oluştur
RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,    // n_estimators
    10,     // max_depth
    2,      // min_samples_split
    1,      // min_samples_leaf
    0,      // max_features (0 = sqrt)
    1,      // bootstrap
    42      // random_state
);

rf->base.fit((Estimator*)rf, X_train, y_train);

double accuracy = rf->base.score((Estimator*)rf, X_test, y_test);
printf("Test doğruluğu: %.4f\n", accuracy);
printf("OOB skoru: %.4f\n", rf->oob_score_);

rf->base.free((Estimator*)rf);
```

### Sinir Ağı

```c
#include "neural_network.h"

// Ağ oluştur: girdi -> 64 -> 32 -> çıktı
size_t layer_sizes[] = {n_features, 64, 32, n_classes};
NeuralNetwork *nn = neural_network_create(layer_sizes, 4, ACTIVATION_RELU);

// Eğitimi yapılandır
nn->learning_rate = 0.001;
nn->epochs = 100;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);

double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);
printf("Sinir ağı doğruluğu: %.4f\n", accuracy);

nn->base.free((Estimator*)nn);
```

### PCA (Boyut İndirgeme)

```c
#include "decomposition.h"

// 2 temel bileşene indirge
PCA *pca = pca_create(2);
pca->base.fit((Estimator*)pca, X, NULL);

// Veriyi dönüştür
Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);

// Açıklanan varyansı kontrol et
const double *evr = pca_explained_variance_ratio(pca);
printf("PC1 varyansın %%%.2f'sini açıklar\n", evr[0] * 100);

// Orijinal veriyi yeniden oluştur
Matrix *X_reconstructed = pca_inverse_transform(pca, X_reduced);

pca->base.free((Estimator*)pca);
```

### Özellik Seçimi

```c
#include "feature_selection.h"

// SelectKBest: F-skoruna göre en iyi 5 özelliği tut
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 5);
selector->base.fit((Estimator*)selector, X, y);

// Seçilen özellik indekslerini al
const int *support = select_k_best_get_support(selector);

// Veriyi sadece seçilen özelliklere dönüştür
Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// VarianceThreshold: Düşük varyanslı özellikleri kaldır
VarianceThreshold *vt = variance_threshold_create(0.1);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

## Örnekler

Kütüphane kapsamlı örnekler içerir:

| Örnek | Açıklama |
|-------|----------|
| `linear_regression_example` | Kapalı form vs gradient descent |
| `logistic_regression_example` | İkili sınıflandırma |
| `knn_example` | k-En Yakın Komşu |
| `kmeans_example` | k-means++ ile kümeleme |
| `estimator_api_example` | Birleşik API gösterimi |
| `cross_validation_example` | K-Fold çapraz doğrulama |
| `pipeline_example` | Ön işleme + model zincirleri |
| `random_forest_example` | Topluluk öğrenmesi |
| `pca_example` | Boyut indirgeme |
| `feature_selection_example` | Özellik önemi ve seçimi |

## Proje Yapısı

```
tinycml/
├── include/              # Genel başlık dosyaları
│   ├── matrix.h          # Matris işlemleri
│   ├── estimator.h       # Birleşik estimator API'si
│   ├── pipeline.h        # Pipeline sistemi
│   ├── validation.h      # Çapraz doğrulama
│   ├── model_selection.h # GridSearchCV
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── ensemble.h        # Rastgele Orman
│   ├── neural_network.h
│   ├── decomposition.h   # PCA
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/                  # Uygulama dosyaları
├── examples/             # Çalıştırılabilir demolar
├── tests/                # Birim testleri
├── data/                 # Örnek veri setleri
└── docs/                 # Dokümantasyon
```

## Lisans

MIT Lisansı - detaylar için LICENSE dosyasına bakın.
