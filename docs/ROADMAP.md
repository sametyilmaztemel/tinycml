# Roadmap / Yol Haritası

[English](#english) | [Türkçe](#türkçe)

---

# English

## Future Enhancements for tinycml

### Neural Networks
- [ ] Feedforward neural network with configurable layers
- [ ] Activation functions (ReLU, sigmoid, tanh, softmax)
- [ ] Backpropagation algorithm
- [ ] Mini-batch gradient descent
- [ ] Dropout regularization

### Decision Trees
- [ ] ID3/C4.5 decision tree algorithm
- [ ] Information gain / Gini impurity splitting criteria
- [ ] Tree pruning (pre and post)
- [ ] Random Forest ensemble

### Additional Algorithms
- [ ] Naive Bayes classifier (Gaussian, Multinomial)
- [ ] Support Vector Machine (linear kernel)
- [ ] Principal Component Analysis (PCA)
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Regularization (L1/L2) for linear models

### Performance Optimizations
- [ ] SIMD optimizations for matrix operations (SSE/AVX)
- [ ] Parallel processing with OpenMP
- [ ] Memory pool for matrix allocations
- [ ] Cache-friendly matrix multiplication (blocked/tiled)

### Features
- [ ] Model serialization (save/load to binary file)
- [ ] Cross-validation utilities (k-fold, stratified)
- [ ] Learning curves (output to CSV for plotting)
- [ ] Early stopping for gradient descent
- [ ] Feature importance scores

### Preprocessing
- [ ] One-hot encoding for categorical variables
- [ ] Polynomial feature expansion
- [ ] Missing value imputation
- [ ] Outlier detection and handling

### Documentation
- [ ] Algorithm theory documentation with math
- [ ] Performance benchmarks
- [ ] Comparison with scikit-learn results
- [ ] Video tutorials

## Contributing

Contributions are welcome! Here's how you can help:

1. **Bug fixes**: Submit issues and pull requests
2. **New algorithms**: Implement algorithms from the roadmap
3. **Documentation**: Improve explanations and examples
4. **Testing**: Add more unit tests and edge cases
5. **Performance**: Profile and optimize critical paths

---

# Türkçe

## tinycml için Gelecek Geliştirmeler

### Sinir Ağları
- [ ] Yapılandırılabilir katmanlara sahip ileri beslemeli sinir ağı
- [ ] Aktivasyon fonksiyonları (ReLU, sigmoid, tanh, softmax)
- [ ] Geri yayılım algoritması
- [ ] Mini-batch gradient descent
- [ ] Dropout düzenlileştirme

### Karar Ağaçları
- [ ] ID3/C4.5 karar ağacı algoritması
- [ ] Bilgi kazancı / Gini safsızlığı bölme kriterleri
- [ ] Ağaç budama (ön ve son)
- [ ] Random Forest topluluk yöntemi

### Ek Algoritmalar
- [ ] Naive Bayes sınıflandırıcı (Gaussian, Multinomial)
- [ ] Destek Vektör Makinesi (lineer çekirdek)
- [ ] Temel Bileşen Analizi (PCA)
- [ ] Lineer Diskriminant Analizi (LDA)
- [ ] Lineer modeller için düzenlileştirme (L1/L2)

### Performans Optimizasyonları
- [ ] Matris işlemleri için SIMD optimizasyonları (SSE/AVX)
- [ ] OpenMP ile paralel işleme
- [ ] Matris ayırmaları için bellek havuzu
- [ ] Önbellek dostu matris çarpımı (bloklu/döşemeli)

### Özellikler
- [ ] Model serileştirme (ikili dosyaya kaydet/yükle)
- [ ] Çapraz doğrulama araçları (k-katlı, katmanlı)
- [ ] Öğrenme eğrileri (grafik çizimi için CSV çıktısı)
- [ ] Gradient descent için erken durdurma
- [ ] Özellik önem puanları

### Ön İşleme
- [ ] Kategorik değişkenler için one-hot kodlama
- [ ] Polinom özellik genişletme
- [ ] Eksik değer doldurma
- [ ] Aykırı değer tespiti ve işleme

### Dokümantasyon
- [ ] Matematik ile algoritma teorisi dokümantasyonu
- [ ] Performans karşılaştırmaları
- [ ] scikit-learn sonuçlarıyla karşılaştırma
- [ ] Video eğitimleri

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! İşte nasıl yardımcı olabileceğiniz:

1. **Hata düzeltmeleri**: Issue ve pull request gönderin
2. **Yeni algoritmalar**: Yol haritasındaki algoritmaları uygulayın
3. **Dokümantasyon**: Açıklamaları ve örnekleri iyileştirin
4. **Test**: Daha fazla birim testi ve uç durum ekleyin
5. **Performans**: Kritik yolları profil edin ve optimize edin
