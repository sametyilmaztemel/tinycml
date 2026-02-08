# tinycml

**Tiny C Machine Learning Library**

[English](#english) | [Türkçe](#türkçe)

---

# English

A tiny, zero-dependency C library for learning machine learning from scratch.

## Overview

tinycml implements fundamental machine learning algorithms in pure C (C11 standard) with zero external dependencies. It's designed for educational purposes, demonstrating how core ML algorithms work under the hood.

## Why tinycml? Advantages Over Modern ML Libraries

### 🎯 Educational Value

| Aspect | tinycml | TensorFlow/PyTorch/scikit-learn |
|--------|---------|--------------------------------|
| **Code Transparency** | Every algorithm is readable, ~100-200 lines each | Thousands of lines, heavy abstractions |
| **Dependencies** | Zero (only standard C library) | Hundreds of packages, complex environments |
| **Understanding** | See exactly how gradient descent works | Black-box functions |
| **Debugging** | Step through with any C debugger | Complex stack traces |

### 🚀 Performance Characteristics

| Feature | tinycml | Python ML Libraries |
|---------|---------|---------------------|
| **Startup Time** | Instant (~1ms) | Seconds (import overhead) |
| **Memory Footprint** | ~50KB binary | 100MB+ with dependencies |
| **No GIL** | True parallelism possible | Python GIL limitations |
| **Embedded Systems** | Runs on microcontrollers | Requires full OS |

### 🔧 Use Cases Where This Library Excels

1. **Learning ML Fundamentals**: Understand the math behind algorithms
2. **Embedded/IoT Devices**: Run ML on resource-constrained hardware
3. **Real-time Systems**: Predictable, low-latency inference
4. **Custom Modifications**: Easy to extend and modify algorithms
5. **No-dependency Environments**: Air-gapped systems, minimal containers

### 📊 When to Use Modern Libraries Instead

- Large-scale training (millions of samples)
- GPU acceleration needed
- Pre-trained models required
- Production systems with established pipelines

## Features

- **Core Linear Algebra**: Matrix operations, vector operations
- **Data Handling**: CSV loading/saving, train/test split, standardization, min-max scaling
- **Supervised Learning**:
  - Linear Regression (closed-form and gradient descent)
  - Logistic Regression (binary classification)
  - k-Nearest Neighbors
- **Unsupervised Learning**:
  - k-Means Clustering
- **Evaluation Metrics**: MSE, RMSE, MAE, Accuracy, Precision, Recall, F1 Score

## Building

### Prerequisites

- C11 compatible compiler (GCC, Clang, MSVC)
- Make (optional, for convenience)
- CMake 3.10+ (alternative build system)

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
./build/examples/logistic_regression_example
./build/examples/knn_example
./build/examples/kmeans_example
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

## Detailed Usage Guide

### Step 1: Include Headers

```c
#include "matrix.h"          // Matrix operations
#include "csv.h"             // Data loading
#include "preprocessing.h"   // Data preprocessing
#include "linear_regression.h"
#include "logistic_regression.h"
#include "knn.h"
#include "kmeans.h"
#include "metrics.h"         // Evaluation metrics
```

### Step 2: Load and Prepare Data

```c
// Load CSV file (1 = has header row)
Matrix *data = csv_load("data/mydata.csv", 1);

// Split into features (X) and target (y)
Matrix *X = matrix_alloc(data->rows, data->cols - 1);
Matrix *y = matrix_alloc(data->rows, 1);

for (size_t i = 0; i < data->rows; i++) {
    for (size_t j = 0; j < data->cols - 1; j++) {
        matrix_set(X, i, j, matrix_get(data, i, j));
    }
    matrix_set(y, i, 0, matrix_get(data, i, data->cols - 1));
}

// Add bias column for regression models
Matrix *X_bias = add_bias_column(X);

// Optional: Standardize features
Scaler *scaler = NULL;
Matrix *X_scaled = standardize_fit_transform(X, &scaler);
```

### Step 3: Train/Test Split

```c
// Split data: 80% train, 20% test
TrainTestSplit split = train_test_split(X_bias, y, 0.2, 42);

// Access split data
Matrix *X_train = split.X_train;
Matrix *X_test = split.X_test;
Matrix *y_train = split.y_train;
Matrix *y_test = split.y_test;
```

### Step 4: Train Models

#### Linear Regression

```c
// Method 1: Closed-form solution (fast, exact)
Matrix *weights = linreg_fit_closed(X_train, y_train);

// Method 2: Gradient descent (iterative)
double learning_rate = 0.01;
int epochs = 1000;
Matrix *weights_gd = linreg_fit_gd(X_train, y_train, learning_rate, epochs);
```

#### Logistic Regression

```c
// Binary classification
Matrix *weights = logreg_fit(X_train, y_train, 0.1, 1000);

// Predict probabilities
Matrix *proba = logreg_predict_proba(X_test, weights);

// Predict class labels (threshold = 0.5)
Matrix *predictions = logreg_predict(X_test, weights, 0.5);
```

#### k-Nearest Neighbors

```c
// Fit model (k=5 neighbors)
KNNModel *knn = knn_fit(X_train, y_train, 5);

// Predict
Matrix *predictions = knn_predict(knn, X_test);

// Don't forget to free
knn_free(knn);
```

#### k-Means Clustering

```c
// Cluster into 3 groups
KMeansModel *kmeans = kmeans_fit(X, 3, 100, 42);

// Get cluster assignments
Matrix *labels = kmeans_predict(kmeans, X);

// Access centroids
for (int c = 0; c < kmeans->k; c++) {
    printf("Centroid %d: ", c);
    for (size_t j = 0; j < kmeans->centroids->cols; j++) {
        printf("%.2f ", matrix_get(kmeans->centroids, c, j));
    }
    printf("\n");
}

kmeans_free(kmeans);
```

### Step 5: Evaluate Models

```c
// Regression metrics
double mse_val = mse(y_test, predictions);
double rmse_val = rmse(y_test, predictions);
double mae_val = mae(y_test, predictions);

// Classification metrics
double acc = accuracy(y_test, predictions);
double prec = precision(y_test, predictions);
double rec = recall(y_test, predictions);
double f1 = f1_score(y_test, predictions);

// Confusion matrix
ConfusionMatrix cm = confusion_matrix(y_test, predictions);
confusion_matrix_print(&cm);
```

### Step 6: Memory Management

**IMPORTANT**: Always free allocated memory!

```c
// Free matrices
matrix_free(data);
matrix_free(X);
matrix_free(y);
matrix_free(X_bias);
matrix_free(weights);
matrix_free(predictions);

// Free train/test split
train_test_split_free(&split);

// Free scalers
scaler_free(scaler);
minmax_scaler_free(mm_scaler);

// Free models
knn_free(knn_model);
kmeans_free(kmeans_model);
```

## API Reference

See [docs/API.md](docs/API.md) for complete API documentation.

## Project Structure

```
tinycml/
├── include/           # Public headers
│   ├── matrix.h       # Matrix operations
│   ├── vector.h       # Vector operations
│   ├── utils.h        # Random numbers, statistics
│   ├── csv.h          # CSV loading/saving
│   ├── preprocessing.h # Data preprocessing
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   └── metrics.h      # Evaluation metrics
├── src/               # Implementation files
├── examples/          # Runnable CLI demos
├── tests/             # Unit tests
├── data/              # Sample CSV datasets
├── docs/              # Documentation
├── .github/workflows/ # CI configuration
├── CMakeLists.txt     # CMake build
├── Makefile           # Direct build
└── README.md          # This file
```

## License

MIT License - see LICENSE file for details.

---

# Türkçe

Makine öğrenmesini sıfırdan öğrenmek için üretim kalitesinde bir C kütüphanesi.

## Genel Bakış

Bu kütüphane, temel makine öğrenmesi algoritmalarını saf C (C11 standardı) ile sıfır harici bağımlılık kullanarak uygular. Eğitim amaçlı tasarlanmış olup, temel ML algoritmalarının nasıl çalıştığını gösterir.

## Neden tinycml? Modern ML Kütüphanelerine Göre Avantajları

### 🎯 Eğitimsel Değer

| Özellik | tinycml | TensorFlow/PyTorch/scikit-learn |
|---------|-------------------|--------------------------------|
| **Kod Şeffaflığı** | Her algoritma okunabilir, ~100-200 satır | Binlerce satır, ağır soyutlamalar |
| **Bağımlılıklar** | Sıfır (sadece standart C kütüphanesi) | Yüzlerce paket, karmaşık ortamlar |
| **Anlama** | Gradient descent'in tam olarak nasıl çalıştığını görün | Kara kutu fonksiyonlar |
| **Hata Ayıklama** | Herhangi bir C debugger ile adım adım izleyin | Karmaşık stack trace'ler |

### 🚀 Performans Özellikleri

| Özellik | tinycml | Python ML Kütüphaneleri |
|---------|-------------------|------------------------|
| **Başlangıç Süresi** | Anlık (~1ms) | Saniyeler (import overhead) |
| **Bellek Kullanımı** | ~50KB binary | 100MB+ bağımlılıklarla |
| **GIL Yok** | Gerçek paralellik mümkün | Python GIL sınırlamaları |
| **Gömülü Sistemler** | Mikrodenetleyicilerde çalışır | Tam işletim sistemi gerektirir |

### 🔧 Bu Kütüphanenin Öne Çıktığı Kullanım Alanları

1. **ML Temellerini Öğrenme**: Algoritmaların arkasındaki matematiği anlayın
2. **Gömülü/IoT Cihazları**: Kaynak kısıtlı donanımlarda ML çalıştırın
3. **Gerçek Zamanlı Sistemler**: Tahmin edilebilir, düşük gecikmeli inference
4. **Özel Modifikasyonlar**: Algoritmaları genişletmek ve değiştirmek kolay
5. **Bağımlılık Gerektirmeyen Ortamlar**: İzole sistemler, minimal container'lar

### 📊 Modern Kütüphanelerin Tercih Edilmesi Gereken Durumlar

- Büyük ölçekli eğitim (milyonlarca örnek)
- GPU hızlandırma gereksinimi
- Önceden eğitilmiş modeller gereksinimi
- Kurulu pipeline'lara sahip üretim sistemleri

## Özellikler

- **Temel Lineer Cebir**: Matris işlemleri, vektör işlemleri
- **Veri İşleme**: CSV yükleme/kaydetme, train/test bölme, standardizasyon, min-max ölçekleme
- **Denetimli Öğrenme**:
  - Lineer Regresyon (kapalı form ve gradient descent)
  - Lojistik Regresyon (ikili sınıflandırma)
  - k-En Yakın Komşu
- **Denetimsiz Öğrenme**:
  - k-Means Kümeleme
- **Değerlendirme Metrikleri**: MSE, RMSE, MAE, Doğruluk, Kesinlik, Duyarlılık, F1 Skoru

## Derleme

### Gereksinimler

- C11 uyumlu derleyici (GCC, Clang, MSVC)
- Make (isteğe bağlı, kolaylık için)
- CMake 3.10+ (alternatif build sistemi)

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
./build/examples/logistic_regression_example
./build/examples/knn_example
./build/examples/kmeans_example
```

## Detaylı Kullanım Rehberi

### Adım 1: Header Dosyalarını Dahil Edin

```c
#include "matrix.h"          // Matris işlemleri
#include "csv.h"             // Veri yükleme
#include "preprocessing.h"   // Veri ön işleme
#include "linear_regression.h"
#include "logistic_regression.h"
#include "knn.h"
#include "kmeans.h"
#include "metrics.h"         // Değerlendirme metrikleri
```

### Adım 2: Veri Yükleme ve Hazırlama

```c
// CSV dosyasını yükle (1 = başlık satırı var)
Matrix *data = csv_load("data/mydata.csv", 1);

// Özellikler (X) ve hedef (y) olarak ayır
Matrix *X = matrix_alloc(data->rows, data->cols - 1);
Matrix *y = matrix_alloc(data->rows, 1);

for (size_t i = 0; i < data->rows; i++) {
    for (size_t j = 0; j < data->cols - 1; j++) {
        matrix_set(X, i, j, matrix_get(data, i, j));
    }
    matrix_set(y, i, 0, matrix_get(data, i, data->cols - 1));
}

// Regresyon modelleri için bias sütunu ekle
Matrix *X_bias = add_bias_column(X);

// İsteğe bağlı: Özellikleri standardize et
Scaler *scaler = NULL;
Matrix *X_scaled = standardize_fit_transform(X, &scaler);
```

### Adım 3: Train/Test Bölmesi

```c
// Veriyi böl: %80 eğitim, %20 test
TrainTestSplit split = train_test_split(X_bias, y, 0.2, 42);

// Bölünmüş verilere eriş
Matrix *X_train = split.X_train;
Matrix *X_test = split.X_test;
Matrix *y_train = split.y_train;
Matrix *y_test = split.y_test;
```

### Adım 4: Modelleri Eğitin

#### Lineer Regresyon

```c
// Yöntem 1: Kapalı form çözümü (hızlı, kesin)
Matrix *weights = linreg_fit_closed(X_train, y_train);

// Yöntem 2: Gradient descent (iteratif)
double learning_rate = 0.01;
int epochs = 1000;
Matrix *weights_gd = linreg_fit_gd(X_train, y_train, learning_rate, epochs);
```

#### Lojistik Regresyon

```c
// İkili sınıflandırma
Matrix *weights = logreg_fit(X_train, y_train, 0.1, 1000);

// Olasılıkları tahmin et
Matrix *proba = logreg_predict_proba(X_test, weights);

// Sınıf etiketlerini tahmin et (eşik = 0.5)
Matrix *predictions = logreg_predict(X_test, weights, 0.5);
```

#### k-En Yakın Komşu

```c
// Modeli fit et (k=5 komşu)
KNNModel *knn = knn_fit(X_train, y_train, 5);

// Tahmin et
Matrix *predictions = knn_predict(knn, X_test);

// Belleği temizlemeyi unutmayın
knn_free(knn);
```

#### k-Means Kümeleme

```c
// 3 gruba kümelendir
KMeansModel *kmeans = kmeans_fit(X, 3, 100, 42);

// Küme atamalarını al
Matrix *labels = kmeans_predict(kmeans, X);

// Merkez noktalarına eriş
for (int c = 0; c < kmeans->k; c++) {
    printf("Merkez %d: ", c);
    for (size_t j = 0; j < kmeans->centroids->cols; j++) {
        printf("%.2f ", matrix_get(kmeans->centroids, c, j));
    }
    printf("\n");
}

kmeans_free(kmeans);
```

### Adım 5: Modelleri Değerlendirin

```c
// Regresyon metrikleri
double mse_val = mse(y_test, predictions);
double rmse_val = rmse(y_test, predictions);
double mae_val = mae(y_test, predictions);

// Sınıflandırma metrikleri
double acc = accuracy(y_test, predictions);
double prec = precision(y_test, predictions);
double rec = recall(y_test, predictions);
double f1 = f1_score(y_test, predictions);

// Karışıklık matrisi
ConfusionMatrix cm = confusion_matrix(y_test, predictions);
confusion_matrix_print(&cm);
```

### Adım 6: Bellek Yönetimi

**ÖNEMLİ**: Ayrılan belleği her zaman serbest bırakın!

```c
// Matrisleri serbest bırak
matrix_free(data);
matrix_free(X);
matrix_free(y);
matrix_free(X_bias);
matrix_free(weights);
matrix_free(predictions);

// Train/test bölmesini serbest bırak
train_test_split_free(&split);

// Ölçekleyicileri serbest bırak
scaler_free(scaler);
minmax_scaler_free(mm_scaler);

// Modelleri serbest bırak
knn_free(knn_model);
kmeans_free(kmeans_model);
```

## API Referansı

Tam API dokümantasyonu için [docs/API_TR.md](docs/API_TR.md) dosyasına bakın.

## Proje Yapısı

```
tinycml/
├── include/           # Genel başlık dosyaları
│   ├── matrix.h       # Matris işlemleri
│   ├── vector.h       # Vektör işlemleri
│   ├── utils.h        # Rastgele sayılar, istatistikler
│   ├── csv.h          # CSV yükleme/kaydetme
│   ├── preprocessing.h # Veri ön işleme
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   └── metrics.h      # Değerlendirme metrikleri
├── src/               # Uygulama dosyaları
├── examples/          # Çalıştırılabilir CLI demoları
├── tests/             # Birim testleri
├── data/              # Örnek CSV veri setleri
├── docs/              # Dokümantasyon
├── .github/workflows/ # CI yapılandırması
├── CMakeLists.txt     # CMake build
├── Makefile           # Doğrudan build
└── README.md          # Bu dosya
```

## Lisans

MIT Lisansı - detaylar için LICENSE dosyasına bakın.
