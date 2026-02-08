# API Referansı

tinycml için tam API dokümantasyonu.

## İçindekiler

1. [Matris İşlemleri](#matris-i̇şlemleri)
2. [Vektör İşlemleri](#vektör-i̇şlemleri)
3. [Yardımcı Fonksiyonlar](#yardımcı-fonksiyonlar)
4. [CSV İşleme](#csv-i̇şleme)
5. [Veri Ön İşleme](#veri-ön-i̇şleme)
6. [Lineer Regresyon](#lineer-regresyon)
7. [Lojistik Regresyon](#lojistik-regresyon)
8. [k-En Yakın Komşu](#k-en-yakın-komşu)
9. [k-Means Kümeleme](#k-means-kümeleme)
10. [Değerlendirme Metrikleri](#değerlendirme-metrikleri)

---

## Matris İşlemleri

**Başlık Dosyası:** `matrix.h`

### Veri Yapısı

```c
typedef struct {
    size_t rows;    // Satır sayısı
    size_t cols;    // Sütun sayısı
    double *data;   // Satır öncelikli veri dizisi
} Matrix;
```

**Bellek Düzeni:** Satır öncelikli (row-major) sıralama. (i, j) konumundaki eleman `data[i * cols + j]` adresindedir.

### Bellek Yönetimi

#### `matrix_alloc`
```c
Matrix* matrix_alloc(size_t rows, size_t cols);
```
Sıfır ile başlatılmış bir matris ayır.

**Parametreler:**
- `rows` - Satır sayısı
- `cols` - Sütun sayısı

**Döndürür:** Ayrılmış matrise işaretçi, başarısızlıkta `NULL`.

**Örnek:**
```c
Matrix *m = matrix_alloc(3, 4);  // 3x4 sıfır matrisi
if (m == NULL) {
    fprintf(stderr, "Bellek ayırma başarısız\n");
}
```

#### `matrix_free`
```c
void matrix_free(Matrix *m);
```
Matris belleğini serbest bırak. `NULL` ile çağrılması güvenlidir.

#### `matrix_copy`
```c
Matrix* matrix_copy(const Matrix *m);
```
Matrisin derin kopyasını oluştur.

### Eleman Erişimi

#### `matrix_get`
```c
double matrix_get(const Matrix *m, size_t i, size_t j);
```
(i, j) konumundaki elemanı al. Debug modunda sınır kontrolü yapar.

#### `matrix_set`
```c
void matrix_set(Matrix *m, size_t i, size_t j, double val);
```
(i, j) konumundaki elemanı ayarla.

### Aritmetik İşlemler

#### `matrix_add`
```c
Matrix* matrix_add(const Matrix *a, const Matrix *b);
```
Eleman bazlı toplama. Boyut uyuşmazlığında `NULL` döndürür.

#### `matrix_sub`
```c
Matrix* matrix_sub(const Matrix *a, const Matrix *b);
```
Eleman bazlı çıkarma.

#### `matrix_mul`
```c
Matrix* matrix_mul(const Matrix *a, const Matrix *b);
```
Eleman bazlı çarpma (Hadamard çarpımı).

#### `matrix_scale`
```c
Matrix* matrix_scale(const Matrix *m, double scalar);
```
Tüm elemanları skaler ile çarp.

#### `matrix_matmul`
```c
Matrix* matrix_matmul(const Matrix *a, const Matrix *b);
```
Matris çarpımı. A(m×k) ve B(k×n) için C(m×n) döndürür.

**Karmaşıklık:** O(m × n × k)

### Dönüşümler

#### `matrix_transpose`
```c
Matrix* matrix_transpose(const Matrix *m);
```
Transpoz matrisi döndür.

### Yardımcılar

#### `matrix_print`
```c
void matrix_print(const Matrix *m);
```
Matrisi stdout'a biçimlendirilmiş olarak yazdır.

#### `matrix_fill`
```c
void matrix_fill(Matrix *m, double val);
```
Tüm elemanları sabit bir değerle doldur.

#### `matrix_identity`
```c
Matrix* matrix_identity(size_t n);
```
n×n birim matris oluştur.

---

## Vektör İşlemleri

**Başlık Dosyası:** `vector.h`

Vektörler (n×1) veya (1×n) matrisler olarak temsil edilir.

### Fonksiyonlar

#### `vector_dot`
```c
double vector_dot(const Matrix *a, const Matrix *b);
```
İç çarpım hesapla. Herhangi bir matris şekliyle çalışır (her ikisini de düzleştirir).

#### `vector_norm`
```c
double vector_norm(const Matrix *v);
```
L2 (Öklid) normu hesapla: `||v|| = sqrt(sum(v_i^2))`.

#### `vector_scale`
```c
Matrix* vector_scale(const Matrix *v, double scalar);
```
Vektörü skaler ile ölçekle (`matrix_scale` için takma ad).

#### `vector_add` / `vector_sub`
```c
Matrix* vector_add(const Matrix *a, const Matrix *b);
Matrix* vector_sub(const Matrix *a, const Matrix *b);
```
Eleman bazlı vektör işlemleri.

---

## Yardımcı Fonksiyonlar

**Başlık Dosyası:** `utils.h`

### Rastgele Sayı Üretimi

#### `rand_seed`
```c
void rand_seed(unsigned int seed);
```
Tekrarlanabilirlik için rastgele sayı üretecini tohumla.

#### `rand_uniform`
```c
double rand_uniform(void);
```
[0, 1) aralığında düzgün dağılımlı rastgele sayı üret.

#### `rand_uniform_range`
```c
double rand_uniform_range(double min, double max);
```
[min, max) aralığında düzgün dağılımlı rastgele sayı üret.

#### `rand_normal`
```c
double rand_normal(void);
```
Box-Muller dönüşümü kullanarak standart normal rastgele sayı üret (ortalama=0, std=1).

#### `rand_normal_params`
```c
double rand_normal_params(double mean, double std);
```
Belirtilen ortalama ve standart sapma ile normal rastgele sayı üret.

### İstatistik

#### `mean`
```c
double mean(const double *data, size_t n);
```
Aritmetik ortalama hesapla.

#### `std_dev`
```c
double std_dev(const double *data, size_t n);
```
Örnek standart sapması hesapla (Bessel düzeltmesi ile).

#### `variance`
```c
double variance(const double *data, size_t n);
```
Örnek varyansı hesapla.

#### `shuffle_indices`
```c
void shuffle_indices(size_t *indices, size_t n);
```
İndeks dizisi için Fisher-Yates karıştırması.

---

## CSV İşleme

**Başlık Dosyası:** `csv.h`

### Fonksiyonlar

#### `csv_load`
```c
Matrix* csv_load(const char *filename, int has_header);
```
CSV dosyasını matrise yükle.

**Parametreler:**
- `filename` - CSV dosya yolu
- `has_header` - İlk satır başlık ise 1 (atlanacak), değilse 0

**Döndürür:** Verili matris, hatada `NULL`.

**Desteklenen formatlar:**
- Virgülle ayrılmış değerler
- Yalnızca sayısal veri (double)
- Unix veya Windows satır sonları

**Örnek:**
```c
// Başlıklı CSV yükle
Matrix *data = csv_load("data/iris.csv", 1);

// Başlıksız CSV yükle
Matrix *data = csv_load("data/sayilar.csv", 0);
```

#### `csv_save`
```c
int csv_save(const Matrix *m, const char *filename);
```
Matrisi CSV dosyasına kaydet.

**Döndürür:** Başarıda 0, hatada -1.

---

## Veri Ön İşleme

**Başlık Dosyası:** `preprocessing.h`

### Train/Test Bölmesi

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

**Parametreler:**
- `X` - Özellik matrisi (n_samples × n_features)
- `y` - Hedef vektörü (n_samples × 1)
- `test_ratio` - Test seti oranı (örn. %20 için 0.2)
- `seed` - Tekrarlanabilirlik için rastgele tohum

**Örnek:**
```c
TrainTestSplit split = train_test_split(X, y, 0.2, 42);
// Eğitim için split.X_train, split.y_train kullan
// Test için split.X_test, split.y_test kullan
train_test_split_free(&split);
```

### Standardizasyon (Z-Skoru)

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

**Formül:** `z = (x - ortalama) / std`

**Örnek:**
```c
Scaler *scaler = NULL;
Matrix *X_scaled = standardize_fit_transform(X_train, &scaler);
Matrix *X_test_scaled = standardize_transform(X_test, scaler);
scaler_free(scaler);
```

### Min-Max Ölçekleme

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

**Formül:** `x_scaled = (x - min) / (max - min)`

Özellikleri [0, 1] aralığına ölçekler.

### Bias Sütunu

```c
Matrix* add_bias_column(const Matrix *X);
```

İlk sütun olarak birler sütunu ekle (regresyon kesim noktası için).

**Örnek:**
```c
// X (n × m) ise, sonuç ilk sütunu 1.0 olan (n × m+1) olur
Matrix *X_bias = add_bias_column(X);
```

---

## Lineer Regresyon

**Başlık Dosyası:** `linear_regression.h`

### Fonksiyonlar

#### `linreg_fit_closed`
```c
Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y);
```
Kapalı form çözümü (normal denklem) kullanarak lineer regresyon fit et.

**Formül:** `w = (X'X)^(-1) X'y`

**Parametreler:**
- `X` - Bias sütunlu özellik matrisi (n × m)
- `y` - Hedef vektörü (n × 1)

**Döndürür:** Ağırlık vektörü (m × 1)

**Not:** X bias sütunu içermelidir. Önce `add_bias_column()` kullanın.

#### `linreg_fit_gd`
```c
Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs);
```
Gradient descent kullanarak lineer regresyon fit et.

**Güncelleme kuralı:** `w = w - lr * X' * (X*w - y) / n`

**Parametreler:**
- `X` - Bias sütunlu özellik matrisi
- `y` - Hedef vektörü
- `lr` - Öğrenme oranı (örn. 0.01)
- `epochs` - İterasyon sayısı

#### `linreg_predict`
```c
Matrix* linreg_predict(const Matrix *X, const Matrix *weights);
```
Hedef değerleri tahmin et: `y_pred = X * weights`

---

## Lojistik Regresyon

**Başlık Dosyası:** `logistic_regression.h`

### Fonksiyonlar

#### `sigmoid`
```c
double sigmoid(double x);
```
Sigmoid fonksiyonu hesapla: `1 / (1 + exp(-x))`

Sayısal kararlılık için taşma koruması içerir.

#### `logreg_fit`
```c
Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs);
```
İkili sınıflandırma için lojistik regresyon fit et.

**Parametreler:**
- `X` - Bias sütunlu özellik matrisi
- `y` - İkili hedef vektörü (0 veya 1 değerleri)
- `lr` - Öğrenme oranı
- `epochs` - İterasyon sayısı

**Gradient:** `gradient = X' * (sigmoid(X*w) - y) / n`

#### `logreg_predict_proba`
```c
Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights);
```
P(y=1|x) olasılıklarını tahmin et.

#### `logreg_predict`
```c
Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold);
```
Eşik kullanarak sınıf etiketleri (0 veya 1) tahmin et (tipik olarak 0.5).

---

## k-En Yakın Komşu

**Başlık Dosyası:** `knn.h`

### Veri Yapısı

```c
typedef struct {
    Matrix *X_train;  // Eğitim özellikleri
    Matrix *y_train;  // Eğitim etiketleri
    int k;            // Komşu sayısı
} KNNModel;
```

### Fonksiyonlar

#### `knn_fit`
```c
KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k);
```
k-NN modeli oluştur (eğitim verilerini saklar).

**Parametreler:**
- `X` - Eğitim özellikleri
- `y` - Eğitim etiketleri (double olarak sınıf indeksleri)
- `k` - Komşu sayısı

#### `knn_predict`
```c
Matrix* knn_predict(const KNNModel *model, const Matrix *X);
```
Çoğunluk oyu kullanarak sınıf etiketleri tahmin et.

**Algoritma:**
1. Her test örneği için, tüm eğitim örneklerine Öklid mesafesini hesapla
2. k en yakın komşuyu bul
3. Komşular arasındaki çoğunluk sınıfını döndür

#### `knn_free`
```c
void knn_free(KNNModel *model);
```
k-NN model belleğini serbest bırak.

---

## k-Means Kümeleme

**Başlık Dosyası:** `kmeans.h`

### Veri Yapısı

```c
typedef struct {
    Matrix *centroids;  // Küme merkezleri (k × n_features)
    int k;              // Küme sayısı
    int max_iter;       // Maksimum iterasyon
} KMeansModel;
```

### Fonksiyonlar

#### `kmeans_fit`
```c
KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed);
```
Lloyd algoritması kullanarak k-Means modeli fit et.

**Parametreler:**
- `X` - Veri matrisi (n_samples × n_features)
- `k` - Küme sayısı
- `max_iter` - Maksimum iterasyon
- `seed` - Merkez başlatma için rastgele tohum

**Algoritma:**
1. Veri noktalarından rastgele k merkez başlat
2. Her noktayı en yakın merkeze ata
3. Merkezleri atanan noktaların ortalaması olarak güncelle
4. Yakınsama veya max_iter'e kadar tekrarla

#### `kmeans_predict`
```c
Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X);
```
Örneklere küme etiketleri ata.

#### `kmeans_free`
```c
void kmeans_free(KMeansModel *model);
```
k-Means model belleğini serbest bırak.

---

## Değerlendirme Metrikleri

**Başlık Dosyası:** `metrics.h`

### Regresyon Metrikleri

#### `mse`
```c
double mse(const Matrix *y_true, const Matrix *y_pred);
```
Ortalama Kare Hatası: `(1/n) * sum((y_true - y_pred)^2)`

#### `rmse`
```c
double rmse(const Matrix *y_true, const Matrix *y_pred);
```
Kök Ortalama Kare Hatası: `sqrt(MSE)`

#### `mae`
```c
double mae(const Matrix *y_true, const Matrix *y_pred);
```
Ortalama Mutlak Hata: `(1/n) * sum(|y_true - y_pred|)`

### Sınıflandırma Metrikleri

#### `accuracy`
```c
double accuracy(const Matrix *y_true, const Matrix *y_pred);
```
Sınıflandırma doğruluğu: `doğru / toplam`

#### `precision`
```c
double precision(const Matrix *y_true, const Matrix *y_pred);
```
İkili sınıflandırma için kesinlik: `TP / (TP + FP)`

#### `recall`
```c
double recall(const Matrix *y_true, const Matrix *y_pred);
```
İkili sınıflandırma için duyarlılık: `TP / (TP + FN)`

#### `f1_score`
```c
double f1_score(const Matrix *y_true, const Matrix *y_pred);
```
F1 Skoru: `2 * (precision * recall) / (precision + recall)`

### Karışıklık Matrisi

```c
typedef struct {
    int tp;  // Gerçek Pozitifler
    int tn;  // Gerçek Negatifler
    int fp;  // Yanlış Pozitifler
    int fn;  // Yanlış Negatifler
} ConfusionMatrix;

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);
void confusion_matrix_print(const ConfusionMatrix *cm);
```

**Örnek:**
```c
ConfusionMatrix cm = confusion_matrix(y_test, predictions);
printf("Doğruluk: %.2f%%\n", (double)(cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn) * 100);
confusion_matrix_print(&cm);
```
