# 00 — Proje Genel Özet

## Projenin Amacı

Bu proje, DonkeyCar çerçevesi üzerine inşa edilmiş bir otonom araç sürüş sistemi geliştirmeyi amaçlamaktadır. Temel hedef, bir araçtaki kameradan alınan görüntüleri kullanarak aracın pist üzerinde otonom olarak sürüş yapabilmesini sağlamaktır.

Klasik yaklaşımlardan farklı olarak bu çalışmada **target-point (hedef nokta) yöntemi** benimsenmiştir. Modelin çıktısı, direksiyon açısı değil, ego-frame koordinat sisteminde ilerleyen bir pistin `(target_x, target_y)` koordinatıdır. Bu koordinat, bir geometrik kontrolcü tarafından direksiyon komutuna dönüştürülür.

## Çözülen Problem

Görüntüden doğrudan direksiyon açısı tahmin eden uçtan-uca (end-to-end) modeller, simülasyon ile gerçek dünya arasındaki görsel farklılıklara (domain gap) karşı kırılgan olmaktadır. Bu çalışmada hedef nokta tahmini yaklaşımı, bu domain gap sorununu azaltmak için tercih edilmiştir.

README.md'de şu şekilde açıklanmaktadır:

> "Bu yaklaşım sim-to-real transferini kolaylaştırır."

## Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|----------------|
| Python 3.11 | Ana programlama dili |
| TensorFlow 2.15.1 / Keras | Derin öğrenme modeli |
| DonkeyCar / gym-donkeycar | Araç kontrol çerçevesi |
| Unity / DonkeySim | Simülasyon ortamı |
| PowerShell | Otomasyon betikleri |
| TFLite | Jetson üzerinde hafif çıkarım |

## Genel Sistem Yapısı

Sistem dört ana bileşenden oluşmaktadır:

1. **Model (CNN):** ~115K parametreli, depthwise separable convolution kullanan verimli bir evrişimsel ağ. Girdi olarak kamera görüntüsü alır, çıktı olarak `(target_x, target_y)` koordinatını verir.
2. **Controller (Geometrik Kontrolcü):** Tahmin edilen hedef noktadan heading error hesaplayarak direksiyon komutuna dönüştürür. Gaz, viraj açısına göre dinamik olarak ayarlanır.
3. **Veri Toplama ve Etiketleme:** DonkeyCar Tub V2 formatında görüntüler toplanır; pist merkez hattı haritalanarak hedef noktalar hesaplanır.
4. **Eğitim Altyapısı:** JSONL manifest formatı kullanılır, veri artırma (augmentation), örnek ağırlıklandırma ve domain randomizasyonu uygulanır.

## Proje Dizin Yapısı (Son Durum)

```
otonom-arac/
├── ai_pipeline/
│   ├── target_point/         # model, controller, dataset, augment, mapping...
│   ├── train.py              # eğitim giriş noktası
│   ├── collect_target_point_data.py
│   ├── build_target_point_labels.py
│   └── evaluate_target_point.py
├── configs/                  # model_01 … model_12 deney konfigürasyonları
├── data/
│   ├── sim_unified_maps/
│   └── artifacts/maps/       # pist merkez hattı haritaları
├── models/                   # eğitilmiş .keras ve .tflite dosyaları
├── scripts/                  # PS1 otomasyon betikleri
├── manage.py                 # araç sürüş / smoke test
├── simulationconfig.py       # tüm parametreler
└── requirements-train.txt
```

## Tez Bağlamında Projenin Önemi

Proje, sim-to-real (simülasyondan gerçeğe) transfer öğrenmesi alanında bir uygulama çalışmasıdır. Hedef nokta tahmiri yaklaşımı; doğrudan direksiyon tahmirinin aksine, görsel domain değişimlerine daha az duyarlı bir ara temsil katmanı oluşturur. Bu, robotik alanda "orta düzey temsil" (mid-level representation) olarak bilinir.

Çalışmada çoklu deneysel aşama (phase) tasarlanmış; model konfigürasyonları sistematik biçimde karşılaştırılmıştır (fixed vs. adaptive lookahead, augmentasyon stratejileri, hybrid veri karışımı).

## Mevcut Repo Yapısından Çıkarılan Genel Sonuçlar

- Proje tek bir branchte (`main`) geliştirilmiştir; paralel deneme branchi görülmemektedir.
- Bazı denemeler (DAgger, scripted expert, temporal model) daha sonra revert ya da silme yoluyla kaldırılmıştır.
- Commit geçmişi birden fazla "deneme → geri alma" döngüsünü yansıtmaktadır.
- Projenin son aşamasında belgeleme (README, COMMANDS.md, SETUP.md) kapsamlı biçimde yazılmıştır; bu, paylaşıma veya tez sunumuna hazırlık sürecini işaret etmektedir.
- macOS ve Windows ikisi için de bağımlılık ve simülatör notları mevcuttur; geliştirme ortamının farklı makinelerde kullanıldığı anlaşılmaktadır.
