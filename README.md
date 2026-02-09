<p align="center">
  <img src="assets/logo.png" alt="tinycml logo" width="300">
</p>

<h1 align="center">tinycml</h1>

<p align="center">
  <strong>Makine Öğrenmesi, Saf C'nin Gücüyle</strong>
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

# English

## What is tinycml?

tinycml is a complete machine learning library written entirely in C. No Python interpreter. No virtual environments. No package managers. No runtime dependencies. Just pure, compiled machine code that runs directly on your processor.

**160KB. 9,700 lines. Every algorithm you can read, understand, and modify.**

## The Philosophy

Modern ML frameworks hide thousands of lines of abstraction behind simple function calls. You call `model.fit()` and magic happens somewhere in a maze of Python, C++, CUDA, and vendor-specific optimizations.

tinycml takes the opposite approach: **complete transparency**. Every matrix multiplication, every gradient calculation, every backpropagation step is right there in readable C code. When you train a neural network with tinycml, you can step through every single operation with a debugger.

## What You Get

### Complete Algorithm Suite

**Regression & Classification**
- Linear Regression — both closed-form (normal equations) and iterative (gradient descent)
- Logistic Regression — binary classification with configurable L2 regularization
- k-Nearest Neighbors — distance-weighted voting for classification and regression
- Naive Bayes — Gaussian likelihood estimation
- Decision Trees — recursive partitioning with Gini impurity or entropy splitting
- Random Forest — bootstrap aggregation with out-of-bag error estimation
- Support Vector Machine — linear kernel with hinge loss optimization
- Neural Networks — fully connected layers, backpropagation, ReLU/sigmoid/tanh/softmax activations

**Unsupervised Learning**
- k-Means Clustering — k-means++ smart initialization, iterative centroid refinement
- Principal Component Analysis — eigendecomposition, variance-based dimensionality reduction, optional whitening

**Feature Engineering**
- SelectKBest — statistical feature ranking (F-test, chi-square, mutual information)
- VarianceThreshold — automatic removal of near-constant features
- StandardScaler — zero mean, unit variance normalization
- MinMaxScaler — bounded range scaling
- OneHotEncoder — categorical to binary expansion
- PolynomialFeatures — interaction and power term generation

**Model Infrastructure**
- Unified Estimator API — consistent `fit`/`predict`/`score` interface across all models
- Pipeline — chain transformers and estimators into single callable objects
- Cross-Validation — k-fold and stratified splitting with aggregated scoring
- GridSearchCV — exhaustive hyperparameter search with cross-validated evaluation
- Model Serialization — binary save/load for trained models

**Evaluation Metrics**
- Regression: MSE, RMSE, MAE, R² coefficient of determination
- Classification: accuracy, precision, recall, F1-score, confusion matrix
- Clustering: inertia, silhouette score

### Technical Specifications

| Property | Value |
|----------|-------|
| Binary Size | ~160KB (static library) |
| Source Lines | ~9,700 |
| External Dependencies | **None** — only C standard library |
| C Standard | C11 |
| Cold Start | <1ms |
| Platforms | Linux, macOS, Windows, embedded systems, microcontrollers |

## Why C?

### Direct Hardware Access
C compiles to native machine code. No interpreter. No JIT compilation. No garbage collector pauses. When you call a function, it executes immediately on the CPU.

### Predictable Performance
Every operation has deterministic timing. No hidden memory allocations. No background threads. No surprises. Critical for real-time systems, embedded devices, and latency-sensitive applications.

### Universal Portability
C runs everywhere: x86, ARM, RISC-V, microcontrollers, mainframes. If it has a C compiler, tinycml runs on it. Cross-compile for your target and deploy.

### Complete Control
You own the memory layout. You control the cache behavior. You decide when allocations happen. When you need to optimize, every byte is accessible.

### Minimal Footprint
160KB for a complete ML toolkit. Deploy on resource-constrained devices where Python environments are impossible. Run inference on microcontrollers with kilobytes of RAM.

## Building

### Requirements
- C11 compatible compiler (GCC 4.7+, Clang 3.1+, MSVC 2015+)
- Make (optional, for convenience)

### Compile

```bash
git clone https://github.com/sametyilmaztemel/tinycml.git
cd tinycml

make          # Build everything
make test     # Run test suite
make clean    # Remove build artifacts
```

### Run Examples

```bash
./build/examples/linear_regression_example
./build/examples/neural_network_example
./build/examples/random_forest_example
./build/examples/pca_example
./build/examples/feature_selection_example
```

## Usage

### Basic Pattern

Every model follows the same interface:

```c
#include "linear_regression.h"

// Create
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Train
model->base.fit((Estimator*)model, X_train, y_train);

// Predict
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Evaluate
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Clean up
model->base.free((Estimator*)model);
```

### Pipeline

Chain preprocessing and models:

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "logistic_regression.h"

Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "model", (Estimator*)logistic_regression_create());

pipe->base.fit((Estimator*)pipe, X_train, y_train);
Matrix *predictions = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

### Neural Network

```c
#include "neural_network.h"

size_t layers[] = {784, 128, 64, 10};  // MNIST architecture
NeuralNetwork *nn = neural_network_create(layers, 4, ACTIVATION_RELU);

nn->learning_rate = 0.001;
nn->epochs = 50;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);
double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);

nn->base.free((Estimator*)nn);
```

### Random Forest

```c
#include "ensemble.h"

RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,   // 100 trees
    15,    // max depth
    2,     // min samples to split
    1,     // min samples per leaf
    0,     // max features (0 = auto)
    1,     // bootstrap
    42     // random seed
);

rf->base.fit((Estimator*)rf, X_train, y_train);
printf("OOB Score: %.4f\n", rf->oob_score_);

rf->base.free((Estimator*)rf);
```

### Cross-Validation

```c
#include "validation.h"

DecisionTreeClassifier *dt = decision_tree_classifier_create();
CrossValResults *cv = cross_val_score((Estimator*)dt, X, y, 5, 1, 42);

printf("Accuracy: %.4f (+/- %.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
dt->base.free((Estimator*)dt);
```

### Hyperparameter Search

```c
#include "model_selection.h"

ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){5, 10, 15}, 3);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

gs->base.fit((Estimator*)gs, X, y);
printf("Best score: %.4f\n", gs->best_score_);

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### PCA

```c
#include "decomposition.h"

PCA *pca = pca_create(2);  // Reduce to 2 dimensions
pca->base.fit((Estimator*)pca, X, NULL);

Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);

const double *variance_ratio = pca_explained_variance_ratio(pca);
printf("Explained variance: %.2f%%, %.2f%%\n",
       variance_ratio[0] * 100, variance_ratio[1] * 100);

pca->base.free((Estimator*)pca);
```

### Feature Selection

```c
#include "feature_selection.h"

// Keep top 10 features by F-score
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 10);
selector->base.fit((Estimator*)selector, X, y);

Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// Remove near-zero variance features
VarianceThreshold *vt = variance_threshold_create(0.01);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

### Model Persistence

```c
// Save
model->base.save((Estimator*)model, "trained_model.bin");

// Load
LinearRegression *loaded = (LinearRegression*)linear_regression_load("trained_model.bin");
```

## Project Structure

```
tinycml/
├── include/           # Header files
│   ├── matrix.h       # Matrix/vector operations
│   ├── estimator.h    # Base estimator interface
│   ├── pipeline.h     # Pipeline system
│   ├── validation.h   # Cross-validation
│   ├── model_selection.h
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── ensemble.h
│   ├── neural_network.h
│   ├── decomposition.h
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/               # Implementation
├── examples/          # Working demos
├── tests/             # Test suite
├── data/              # Sample datasets
└── docs/              # API documentation
```

## License

MIT License

---

# Türkçe

## tinycml Nedir?

tinycml, tamamen C ile yazılmış eksiksiz bir makine öğrenmesi kütüphanesidir. Python yorumlayıcısı yok. Sanal ortam yok. Paket yöneticisi yok. Çalışma zamanı bağımlılığı yok. Sadece saf, derlenmiş makine kodu — doğrudan işlemcinizde çalışır.

**160KB. 9.700 satır. Okuyabileceğiniz, anlayabileceğiniz ve değiştirebileceğiniz her algoritma.**

## Felsefe

Modern ML çerçeveleri, basit fonksiyon çağrılarının arkasına binlerce satır soyutlama gizler. `model.fit()` dersiniz ve Python, C++, CUDA ve üreticiye özel optimizasyonlar labirentinde bir yerlerde sihir gerçekleşir.

tinycml tam tersi yaklaşımı benimser: **tam şeffaflık**. Her matris çarpımı, her gradyan hesaplaması, her geri yayılım adımı okunabilir C kodunda tam orada. tinycml ile bir sinir ağı eğittiğinizde, bir hata ayıklayıcı ile her bir işlemi adım adım izleyebilirsiniz.

## Ne Elde Edersiniz

### Eksiksiz Algoritma Seti

**Regresyon ve Sınıflandırma**
- Lineer Regresyon — hem kapalı form (normal denklemler) hem de yinelemeli (gradyan iniş)
- Lojistik Regresyon — yapılandırılabilir L2 düzenlileştirme ile ikili sınıflandırma
- k-En Yakın Komşu — sınıflandırma ve regresyon için mesafe ağırlıklı oylama
- Naive Bayes — Gaussian olasılık tahmini
- Karar Ağaçları — Gini safsızlığı veya entropi bölünmesi ile özyinelemeli bölümleme
- Rastgele Orman — torba dışı hata tahmini ile bootstrap toplama
- Destek Vektör Makinesi — menteşe kaybı optimizasyonu ile lineer çekirdek
- Sinir Ağları — tam bağlı katmanlar, geri yayılım, ReLU/sigmoid/tanh/softmax aktivasyonları

**Denetimsiz Öğrenme**
- k-Means Kümeleme — k-means++ akıllı başlatma, yinelemeli merkez iyileştirme
- Temel Bileşen Analizi — özdeğer ayrışımı, varyans tabanlı boyut indirgeme, isteğe bağlı beyazlatma

**Özellik Mühendisliği**
- SelectKBest — istatistiksel özellik sıralaması (F-testi, ki-kare, karşılıklı bilgi)
- VarianceThreshold — neredeyse sabit özelliklerin otomatik kaldırılması
- StandardScaler — sıfır ortalama, birim varyans normalizasyonu
- MinMaxScaler — sınırlı aralık ölçekleme
- OneHotEncoder — kategorikten ikiliye genişleme
- PolynomialFeatures — etkileşim ve kuvvet terimi üretimi

**Model Altyapısı**
- Birleşik Estimator API — tüm modellerde tutarlı `fit`/`predict`/`score` arayüzü
- Pipeline — dönüştürücüleri ve tahmincileri tek çağrılabilir nesnelere zincirleyin
- Çapraz Doğrulama — toplu puanlama ile k-katlı ve tabakalı bölme
- GridSearchCV — çapraz doğrulamalı değerlendirme ile kapsamlı hiperparametre araması
- Model Serileştirme — eğitilmiş modeller için ikili kaydet/yükle

**Değerlendirme Metrikleri**
- Regresyon: MSE, RMSE, MAE, R² belirleme katsayısı
- Sınıflandırma: doğruluk, kesinlik, duyarlılık, F1-skoru, karışıklık matrisi
- Kümeleme: atalet, silhouette skoru

### Teknik Özellikler

| Özellik | Değer |
|---------|-------|
| Binary Boyutu | ~160KB (statik kütüphane) |
| Kaynak Satırı | ~9.700 |
| Harici Bağımlılık | **Hiç** — sadece C standart kütüphanesi |
| C Standardı | C11 |
| Soğuk Başlangıç | <1ms |
| Platformlar | Linux, macOS, Windows, gömülü sistemler, mikrodenetleyiciler |

## Neden C?

### Doğrudan Donanım Erişimi
C, yerel makine koduna derlenir. Yorumlayıcı yok. JIT derlemesi yok. Çöp toplayıcı duraklamaları yok. Bir fonksiyon çağırdığınızda, CPU'da anında çalışır.

### Öngörülebilir Performans
Her işlem deterministik zamanlama ile gerçekleşir. Gizli bellek tahsisi yok. Arka plan iş parçacıkları yok. Sürpriz yok. Gerçek zamanlı sistemler, gömülü cihazlar ve gecikmeye duyarlı uygulamalar için kritik.

### Evrensel Taşınabilirlik
C her yerde çalışır: x86, ARM, RISC-V, mikrodenetleyiciler, ana bilgisayarlar. Bir C derleyicisi varsa, tinycml üzerinde çalışır. Hedefiniz için çapraz derleyin ve dağıtın.

### Tam Kontrol
Bellek düzenine sahipsiniz. Önbellek davranışını kontrol edersiniz. Tahsislerin ne zaman olacağına siz karar verirsiniz. Optimize etmeniz gerektiğinde, her bayta erişilebilir.

### Minimal Ayak İzi
Eksiksiz bir ML araç seti için 160KB. Python ortamlarının imkansız olduğu kaynak kısıtlı cihazlarda dağıtın. Kilobayt RAM'li mikrodenetleyicilerde çıkarım çalıştırın.

## Derleme

### Gereksinimler
- C11 uyumlu derleyici (GCC 4.7+, Clang 3.1+, MSVC 2015+)
- Make (isteğe bağlı)

### Derle

```bash
git clone https://github.com/sametyilmaztemel/tinycml.git
cd tinycml

make          # Her şeyi derle
make test     # Test paketini çalıştır
make clean    # Derleme çıktılarını temizle
```

### Örnekleri Çalıştır

```bash
./build/examples/linear_regression_example
./build/examples/neural_network_example
./build/examples/random_forest_example
./build/examples/pca_example
./build/examples/feature_selection_example
```

## Kullanım

### Temel Desen

Her model aynı arayüzü takip eder:

```c
#include "linear_regression.h"

// Oluştur
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Eğit
model->base.fit((Estimator*)model, X_train, y_train);

// Tahmin et
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Değerlendir
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Temizle
model->base.free((Estimator*)model);
```

### Pipeline

Ön işleme ve modelleri zincirleyin:

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "logistic_regression.h"

Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "model", (Estimator*)logistic_regression_create());

pipe->base.fit((Estimator*)pipe, X_train, y_train);
Matrix *predictions = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

### Sinir Ağı

```c
#include "neural_network.h"

size_t layers[] = {784, 128, 64, 10};  // MNIST mimarisi
NeuralNetwork *nn = neural_network_create(layers, 4, ACTIVATION_RELU);

nn->learning_rate = 0.001;
nn->epochs = 50;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);
double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);

nn->base.free((Estimator*)nn);
```

### Rastgele Orman

```c
#include "ensemble.h"

RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,   // 100 ağaç
    15,    // maksimum derinlik
    2,     // bölmek için minimum örnek
    1,     // yaprak başına minimum örnek
    0,     // maksimum özellik (0 = otomatik)
    1,     // bootstrap
    42     // rastgele tohum
);

rf->base.fit((Estimator*)rf, X_train, y_train);
printf("OOB Skoru: %.4f\n", rf->oob_score_);

rf->base.free((Estimator*)rf);
```

### Çapraz Doğrulama

```c
#include "validation.h"

DecisionTreeClassifier *dt = decision_tree_classifier_create();
CrossValResults *cv = cross_val_score((Estimator*)dt, X, y, 5, 1, 42);

printf("Doğruluk: %.4f (+/- %.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
dt->base.free((Estimator*)dt);
```

### Hiperparametre Araması

```c
#include "model_selection.h"

ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){5, 10, 15}, 3);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

gs->base.fit((Estimator*)gs, X, y);
printf("En iyi skor: %.4f\n", gs->best_score_);

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### PCA

```c
#include "decomposition.h"

PCA *pca = pca_create(2);  // 2 boyuta indirge
pca->base.fit((Estimator*)pca, X, NULL);

Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);

const double *variance_ratio = pca_explained_variance_ratio(pca);
printf("Açıklanan varyans: %.2f%%, %.2f%%\n",
       variance_ratio[0] * 100, variance_ratio[1] * 100);

pca->base.free((Estimator*)pca);
```

### Özellik Seçimi

```c
#include "feature_selection.h"

// F-skoruna göre en iyi 10 özelliği tut
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 10);
selector->base.fit((Estimator*)selector, X, y);

Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// Sıfıra yakın varyanslı özellikleri kaldır
VarianceThreshold *vt = variance_threshold_create(0.01);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

### Model Kalıcılığı

```c
// Kaydet
model->base.save((Estimator*)model, "trained_model.bin");

// Yükle
LinearRegression *loaded = (LinearRegression*)linear_regression_load("trained_model.bin");
```

## Proje Yapısı

```
tinycml/
├── include/           # Başlık dosyaları
│   ├── matrix.h       # Matris/vektör işlemleri
│   ├── estimator.h    # Temel estimator arayüzü
│   ├── pipeline.h     # Pipeline sistemi
│   ├── validation.h   # Çapraz doğrulama
│   ├── model_selection.h
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── ensemble.h
│   ├── neural_network.h
│   ├── decomposition.h
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/               # Uygulama
├── examples/          # Çalışan demolar
├── tests/             # Test paketi
├── data/              # Örnek veri setleri
└── docs/              # API dokümantasyonu
```

## Lisans

MIT Lisansı
