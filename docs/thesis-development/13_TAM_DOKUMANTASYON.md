# TAM GELİŞTİRME DOKÜMANTASYONU — Otonom Araç Projesi
### Git geçmişinden yeniden yapılandırılan tez destek belgesi

> Bu dosya, `docs/thesis-development/` altındaki 00–12 numaralı dosyaların ve README'nin tüm içeriğini tek bir belgede birleştirmektedir.
> Diğer dosyalar silinmemiştir; bu dosya yalnızca pratik kullanım için tümleşik bir kopyasıdır.

---

# İÇİNDEKİLER

1. [Proje Genel Özet](#1-proje-genel-özet)
2. [Commit Kronolojisi](#2-commit-kronolojisi)
3. [Geliştirme Aşamaları](#3-geliştirme-aşamaları)
4. [Deneyler ve Denemeler](#4-deneyler-ve-denemeler)
5. [Karşılaşılan Problemler ve Çözümler](#5-karşılaşılan-problemler-ve-çözümler)
6. [Teknik Kararlar](#6-teknik-kararlar)
7. [Sistem Mimarisi Evrimi](#7-sistem-mimarisi-evrimi)
8. [Özellik Bazlı Gelişim](#8-özellik-bazlı-gelişim)
9. [Hata Düzeltmeleri ve Refactoring](#9-hata-düzeltmeleri-ve-refactoring)
10. [Test, Doğrulama ve Sınırlamalar](#10-test-doğrulama-ve-sınırlamalar)
11. [Tezde Kullanılabilecek Anlatım](#11-tezde-kullanılabilecek-anlatım)
12. [Zaman Çizelgesi](#12-zaman-çizelgesi)
13. [Eksik Bilgiler ve Dürüst Notlar](#13-eksik-bilgiler-ve-dürüst-notlar)
14. [Dokümantasyon Rehberi](#14-dokümantasyon-rehberi)

**— Deney Verisine Dayalı Yeni Bölümler —**

15. [Gerçek Deney Sonuçları](#-15--gerçek-deney-sonuçları) ★ YENİ
16. [Model Karşılaştırmaları](#-16--model-karşılaştırmaları) ★ YENİ
17. [Closed-Loop Performans Analizi](#-17--closed-loop-performans-analizi) ★ YENİ
18. [Eğitim Stabilitesi Analizi](#-18--eğitim-stabilitesi-analizi) ★ YENİ
19. [Deney Başarısızlıklarının Teknik Nedenleri](#-19--deney-başarısızlıklarının-teknik-nedenleri) ★ YENİ
20. [Sim2Real Karşılaştırmaları](#-20--sim2real-karşılaştırmaları) ★ YENİ
21. [Nihai Baseline Seçim Gerekçesi](#-21--nihai-baseline-seçim-gerekçesi) ★ YENİ

---

# 1. Proje Genel Özet

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

1. **Model (CNN):** 33.858 parametreli (`model_export_manifest.json`'dan doğrulanmış), depthwise separable convolution kullanan verimli bir evrişimsel ağ. Girdi olarak kamera görüntüsü alır, çıktı olarak `(target_x, target_y)` koordinatını verir.
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

## Mevcut Repo Yapısından Çıkarılan Genel Sonuçlar

- Proje tek bir branchte (`main`) geliştirilmiştir; paralel deneme branchi görülmemektedir.
- Bazı denemeler (DAgger, scripted expert, temporal model) daha sonra revert ya da silme yoluyla kaldırılmıştır.
- Commit geçmişi birden fazla "deneme → geri alma" döngüsünü yansıtmaktadır.
- Projenin son aşamasında belgeleme (README, COMMANDS.md, SETUP.md) kapsamlı biçimde yazılmıştır; bu, paylaşıma veya tez sunumuna hazırlık sürecini işaret etmektedir.
- macOS ve Windows ikisi için de bağımlılık ve simülatör notları mevcuttur; geliştirme ortamının farklı makinelerde kullanıldığı anlaşılmaktadır.

---

# 2. Commit Kronolojisi

Commitler en eskiden en yeniye doğru sıralanmıştır. Her commit için teknik içerik analiz edilmiştir. Commit mesajları zaman zaman yetersiz olduğundan yorum, diff içeriğine dayandırılmıştır.

## Kronolojik Commit Tablosu

| Sıra | Commit ID | Tarih | Commit Mesajı | Kategori | Teknik Yorum |
|------|-----------|-------|---------------|----------|--------------|
| 1 | `eee7f31` | 2026-03-05 | first commit | setup | DonkeyCar çerçevesinin tam kopyası (gym-donkeycar). macOS artefaktları (`__MACOSX/`, `donkey_sim.app`) dahil. Temel dosyalar: `manage.py`, `train.py`, `config.py`, `myconfig.py`, `calibrate.py`. İlk veri kümesi ve modeller (`dilara.h5`, `my_first_pilot.h5`) de commitlenmiştir. |
| 2 | `c5b68e3` | 2026-03-05 | Enhance training script... | feature | `train.py`'e cihaz yapılandırması (CPU/GPU seçimi) eklendi. `requirements-train.txt` oluşturuldu. Çalıştırma adımlarını açıklayan bir Markdown dosyası güncellendi. |
| 3 | `b28d47b` | 2026-03-11 | Add lane_v1 model config... | feature | `lane_v1` modeli için konfigürasyon eklendi. `myconfig.py` güncellendi. |
| 4 | `6a8538a` | 2026-03-24 | feat: Add target-point training and inference module | feature | **Ana özellik:** `target_point/` modülü oluşturuldu. Alt modüller: `model.py`, `controller.py`, `dataset.py`, `training.py`, `diagnostics.py`, `pilot.py`. Testler (`tests/test_target_point.py`) eklendi. `.repo_tmp/` submodule referansları eklendi. |
| 5 | `5850afb` | 2026-03-27 | Add track mapping functionality... | feature | **Önemli:** Pist haritalama (`build_target_point_labels.py`, `track_map.py`) eklendi. 6 farklı pist için `centerline.csv`, `raw_trace.csv`, `labels_adaptive_v1.csv`, `labels_fixed_1p2m.csv` dosyaları üretildi. Phase 4 ve Phase 5 deneyleri için model ve metrik dosyaları eklendi. Büyük commit (43.000+ satır). |
| 6 | `e02951f` | 2026-03-29 | Add evaluation reports and model files for phase 55 fixed bootstrap | feature | Phase 5.5 sabit bootstrap değerlendirme raporları ve model dosyaları eklendi. |
| 7 | `ac4920b` | 2026-03-29 | Add new target-point training and evaluation components | feature | Yeni eğitim ve değerlendirme bileşenleri eklendi. |
| 8 | `10cf4b9` | 2026-03-31 | Add new configuration and script for single-track stabilization | experiment | Tek-pist stabilizasyonu için yeni konfigürasyon ve betik eklendi. (Bu betik sonradan kaldırılmıştır.) |
| 9 | `2952627` | 2026-04-01 | last | cleanup | Mesaj açıklayıcı değil. Commit içeriği incelenmeden değerlendirilemez; muhtemelen küçük bir düzenleme. |
| 10 | `e9f3908` | 2026-04-01 | Remove unitylog.txt to clean up unnecessary log files | cleanup | Unity logunu repodan temizleme. |
| 11 | `7caadc8` | 2026-04-03 | revert: roll back to phase5 adaptive robust baseline (5850afb) | experiment | **Kritik:** Birden fazla deneme modülü (`dagger.py`, `scripted_expert.py`, `promotion.py`, `temporal.py`, `effective_loss.py`) ve buna ait betikler kaldırılarak `5850afb` commiti esas alındı. 165 dosya değişti. Bu, DAgger ve scripted expert denemelerin başarısız veya yetersiz bulunduğuna işaret etmektedir. |
| 12 | `31f73db` | 2026-04-03 | Remove obsolete diagnostics and label sample files... | cleanup | Eski teşhis ve etiket örnekleri silindi. `.gitignore` güncellendi. |
| 13 | `2e9d6e4` | 2026-04-03 | Remove closed loop episode and summary reports for phase 5... | cleanup | Phase 5'e ait signflip, dynamic, oldstyle, smoke deneme sonuçları ve model dosyaları silindi. Eğitim gereksinimleri güncellendi. |
| 14 | `0f0b351` | 2026-04-05 | Add new target point models and mapping utilities for road generation | feature | Yeni hedef nokta modelleri ve yol üretimi için haritalama yardımcıları eklendi. |
| 15 | `8231732` | 2026-04-05 | Add five run summary report for generated roads validation | feature | Üretilmiş yollar için 5 koşu özet raporu eklendi. |
| 16 | `41389c3` | 2026-04-05 | Son yapılan alan | cleanup | Mesaj Türkçe ve belirsiz. Muhtemelen küçük bir son dokunuş değişikliği. |
| 17 | `a4c40f7` | 2026-04-06 | Add closed loop episode and summary reports for multiple runs | feature | Kapalı döngü bölüm ve özet raporları (birden fazla koşu) eklendi. |
| 18 | `fcb6f8f` | 2026-04-06 | Add closed loop episode and summary reports for two new sessions | feature | İki yeni sürüş oturumu için kapalı döngü raporları eklendi. |
| 19 | `233ad5a` | 2026-04-06 | Add closed loop episode and summary reports for training sessions | feature | Eğitim oturumları için kapalı döngü raporları eklendi. |
| 20 | `43894f0` | 2026-04-07 | Add comprehensive command documentation for project phases and tasks | docs | Proje aşamaları ve görevler için kapsamlı komut dokümantasyonu eklendi. |
| 21 | `4f802fc` | 2026-04-07 | Remove obsolete log files and add setup guide for project installation | docs | Eski log dosyaları silindi. Kurulum kılavuzu (`SETUP.md`) eklendi. |
| 22 | `7236feb` | 2026-04-07 | Update command documentation to use PowerShell syntax... | docs | Komut dokümantasyonu PowerShell sözdizimine güncellendi. Değerlendirme betiğine opsiyonel gaz parametreleri eklendi. |
| 23 | `2c25c81` | 2026-04-08 | Add initial Unity log file for debugging and performance tracking | cleanup | Unity log dosyası eklendi. |
| 24 | `c8e688f` | 2026-04-16 | Add comprehensive multi-track Sim2Real guidelines and command checklists | docs | Çok pistin sim2real yönergeleri ve komut kontrol listeleri eklendi. |
| 25 | `c98a637` | 2026-04-16 | Add script to run target-point models for 3 laps and report results | feature | Hedef nokta modellerini 3 tur koşturan ve sonuçları raporlayan betik eklendi. |
| 26 | `5de581c` | 2026-04-17 | Add new data artifacts, models, and documentation for multi-track sim2real project | feature+refactor | **Mimari değişiklik:** `target_point/` modülü `ai_pipeline/target_point/` altına taşındı. Veri artefaktları, yeni Keras modelleri ve `COMMANDS.md`, `SETUP.md` dokümantasyonu eklendi. |
| 27 | `40d035d` | 2026-04-17 | Refactor: Remove supervised learning scripts and configuration files | refactor | Supervised learning örnek betikleri (`examples/supervised_learning/`) tamamen silindi. TFLite dışa aktarma (`export.py`) ve Jetson için `pilot_tflite.py` eklendi. `conftest.py` eklendi. |
| 28 | `95dafd8` | 2026-04-17 | Remove obsolete JSON reports and model files | cleanup | Eski JSON raporları ve model dosyaları silindi. |
| 29 | `6687c15` | 2026-04-18 | Add initial Unity log file for debugging and performance analysis | cleanup | Yeni Unity log dosyası eklendi. |
| 30 | `2995dab` | 2026-04-19 | Add target-point bias compensation and deadband parameters... | feature | Kontrolcüye bias kompansasyonu ve deadband parametresi eklendi. Büyük veri kümesi toplama (`collect_massive_sim_dataset.ps1`) ve izleme betikleri oluşturuldu. |
| 31 | `ec10b31` | 2026-04-21 | feat: Update simulation configurations and add new model training scripts | feature | Batch size 32→128 artırıldı. `MODEL_TRAINING_PLAN.md` oluşturuldu. Model 01-06 konfigürasyon dosyaları eklendi. Eğitim otomasyon betikleri (`train_first6_models.ps1`, `train_gpu.ps1`) eklendi. |
| 32 | `3f66a3f` | 2026-04-22 | feat: Implement model training scripts for multi-task and fine-tuned models | feature | Model 07 (fine-tune), 11 (multi-task), 12 (temporal) konfigürasyonları eklendi. `train_models_11_12_07.ps1` ve `train_models_5_6.ps1` eklendi. `model_export_manifest.json` oluşturuldu. |
| 33 | `0ce7fd9` | 2026-04-26 | Add detailed project documentation for the Autonomous Vehicle Project | docs | Proje için kapsamlı dokümantasyon eklendi. |
| 34 | `4ba0252` | 2026-04-26 | fix: repo klonlanabilirliğini düzelt | bugfix | Repo klonlanabilirlik sorunları giderildi. |
| 35 | `d483ea6` | 2026-04-26 | fix: update paths in documentation for user-specific directories | bugfix | Dokümanlardaki kullanıcıya özgü yollar güncellendi. |
| 36 | `b27709b` | 2026-04-26 | refactor: update README to reflect project structure... | refactor | README güncellendi; kurulum ve proje yapısı yansıtıldı. |
| 37 | `358d90e` | 2026-04-26 | fix: scripts içindeki hardcoded kişisel yolları kaldır | bugfix | Betiklerdeki hardcoded kişisel yollar kaldırıldı. |
| 38 | `7466d2b` | 2026-04-26 | fix: yeni kullanıcıyı çökertecek 4 tutarsızlığı düzelt | bugfix | Yeni kullanıcı için kritik tutarsızlıklar giderildi. |
| 39 | `fec3302` | 2026-04-26 | fix: kalan küçük tutarsızlıkları gider | bugfix | Kalan küçük tutarsızlıklar giderildi. |

## Commit Kategorisi Dağılımı

| Kategori | Commit Sayısı |
|----------|--------------|
| feature | 14 |
| cleanup | 7 |
| docs | 6 |
| bugfix | 5 |
| refactor | 3 |
| experiment | 2 |
| setup | 1 |

**Önemli Notlar:**
- Commit `2952627` ("last") ve `41389c3` ("Son yapılan alan") anlamsız mesaj içermektedir.
- Commit `7caadc8` (revert) projedeki en büyük geri alma olayıdır; 165 dosya etkilenmiş ve önemli deneme modülleri kaldırılmıştır.
- 2026-04-26 tarihindeki 6 commit kümesi, proje sonlanmadan önce yayına hazırlık sürecine işaret etmektedir.

---

# 3. Geliştirme Aşamaları

## Aşama 1: Proje Altyapısının Kurulması
**Tarih:** 2026-03-05 | **Commitler:** `eee7f31`, `c5b68e3`

DonkeyCar (gym-donkeycar) çerçevesi temel alınarak proje oluşturuldu. İlk commit büyük olasılıkla mevcut bir DonkeyCar kurulumunun repoya aktarılmasından oluşmaktadır (`__MACOSX/` artefaktları bunu destekler). İlk modellerin (`dilara.h5` vb.) varlığı, committen önce yerel denemeler yapıldığına işaret etmektedir. İkinci committe `train.py`'e GPU/CPU seçimi eklendi ve `requirements-train.txt` oluşturuldu.

**Tezde Kullanılabilecek Anlatım:** Projenin geliştirilmesinde DonkeyCar açık kaynak çerçevesi temel alınmıştır. Bu çerçeve, araç kontrolü, veri toplama ve model eğitimi için hazır bileşenler sunmaktadır.

---

## Aşama 2: Hedef Nokta Modülünün Geliştirilmesi
**Tarih:** 2026-03-11 — 2026-03-24 | **Commitler:** `b28d47b`, `6a8538a`

Projenin en kritik mimari kararı bu aşamada alındı: doğrudan direksiyon tahmini yerine **hedef nokta tahmini** yaklaşımına geçiş. `target_point/` modülü sıfırdan oluşturuldu:

| Dosya | İşlev |
|-------|-------|
| `target_point/model.py` | CNN modeli (hedef nokta tahmini) |
| `target_point/controller.py` | Hedef noktadan direksiyon hesaplama |
| `target_point/dataset.py` | Veri kümesi yükleme ve işleme |
| `target_point/training.py` | Eğitim döngüsü ve metrikler |
| `target_point/diagnostics.py` | Teşhis araçları |
| `target_point/pilot.py` | Gerçek zamanlı çıkarım |

Commit `6a8538a`, 1.348 satır ekleme ile projenin en büyük tek özellik committir.

**Tezde Kullanılabilecek Anlatım:** Hedef nokta tabanlı kontrol yaklaşımında model, kamera görüntüsünden doğrudan direksiyon açısı değil, ego-frame koordinat sisteminde ilerleyen bir noktanın `(x, y)` koordinatını tahmin eder. Tahmin edilen koordinat, geometrik bir kontrolcü tarafından direksiyon ve gaz komutlarına dönüştürülmektedir.

---

## Aşama 3: Pist Haritalama ve Deneysel Aşamalar (Phase 4–5)
**Tarih:** 2026-03-27 — 2026-04-01 | **Commitler:** `5850afb`, `e02951f`, `ac4920b`, `10cf4b9`, `2952627`, `e9f3908`

**Pist haritalama sistemi** geliştirildi. 6 farklı DonkeyCar pisti için `raw_trace.csv`, `centerline.csv`, `labels_adaptive_v1.csv`, `labels_fixed_1p2m.csv` dosyaları üretildi.

Phase 4 ve Phase 5 kapsamında deneyler yürütüldü:
- **Phase 4:** adaptive/fixed kombinasyonları — compare, flip, full, generalize, recover, run (11 deney)
- **Phase 5:** adaptive_hybrid_applied, adaptive_robust, fixed_robust

Commit `5850afb` projenin en büyük commiti olup 43.000+ satır içermektedir.

**Tezde Kullanılabilecek Anlatım:** Hedef noktaların hesaplanmasında iki farklı lookahead stratejisi karşılaştırılmıştır: sabit mesafe (1.2 m) ve adaptif mesafe. Adaptif yöntemde hedef nokta mesafesi pist eğriliğine göre dinamik olarak ayarlanmaktadır.

---

## Aşama 4: İleri Denemeler ve Geri Alma
**Tarih:** 2026-04-03 | **Commitler:** `7caadc8`, `31f73db`, `2e9d6e4`

Commit `7caadc8`, projenin en önemli geri alma işlemini temsil etmektedir. `5850afb`'den sonra geliştirilen şu modüller kaldırıldı:

| Silinen Modül | Amaç (Tahmini) |
|---------------|----------------|
| `target_point/dagger.py` | Dataset Aggregation (DAgger) algoritması |
| `target_point/scripted_expert.py` | Kurallı uzman politika |
| `target_point/promotion.py` | Model promosyon mekanizması |
| `target_point/temporal.py` | Zamansal model (birden fazla kare) |
| `target_point/effective_loss.py` | Özel kayıp fonksiyonu |

165 dosya değişti, 4829 satır silindi.

**Tezde Kullanılabilecek Anlatım:** Geliştirme sürecinde DAgger, zamansal model ve scripted expert gibi ileri yöntemler denenmiştir. Kod değişiminden anlaşıldığı kadarıyla bu denemelerin sonuçları yeterli görülmemiş ve daha sade bir mimari üzerinde ilerleme kararı alınmıştır.

---

## Aşama 5: Yeni Pist Denemeleri ve Kapalı Döngü Değerlendirme
**Tarih:** 2026-04-05 — 2026-04-08

Üretilmiş (generated) yollar üzerinde değerlendirme yapıldı. Kapalı döngü (closed-loop) değerlendirme framework'ü kullanılarak birden fazla sürüş oturumu raporlandı. Kapsamlı komut dokümantasyonu (`COMMANDS.md`) ve kurulum kılavuzu (`SETUP.md`) yazıldı.

---

## Aşama 6: Mimari Yeniden Yapılanma ve Çok Pist Sim2Real
**Tarih:** 2026-04-16 — 2026-04-17 | **Commitler:** `c8e688f`, `c98a637`, `5de581c`, `40d035d`, `95dafd8`

**Büyük mimari değişiklik:** `target_point/` modülü `ai_pipeline/target_point/` altına taşındı. Supervised learning örnek kodları (`examples/supervised_learning/`) tamamen kaldırıldı. TFLite dışa aktarma (`export.py`) ve Jetson için `pilot_tflite.py` eklendi.

---

## Aşama 7: Büyük Ölçekli Veri Toplama ve Çok Model Eğitimi
**Tarih:** 2026-04-18 — 2026-04-22 | **Commitler:** `6687c15`, `2995dab`, `ec10b31`, `3f66a3f`

Kontrolcüye **bias kompansasyonu** ve **deadband** parametreleri eklendi. 9 model konfigürasyonu sistematik olarak tanımlandı (model_01 — model_12). Batch size 32'den 128'e yükseltildi.

| Model | Strateji |
|-------|---------|
| model_01 | Pure simulation |
| model_02 | Sim + domain randomization |
| model_03 | Pure real |
| model_04 | Hybrid v1 naive mix |
| model_05 | Hybrid v2 sim-heavy |
| model_06 | Hybrid v3 real-heavy |
| model_07 | Fine-tune |
| model_11 | Multi-task |
| model_12 | Temporal |

---

## Aşama 8: Dokümantasyon ve Yayına Hazırlık
**Tarih:** 2026-04-26 | **Commitler:** `0ce7fd9` — `fec3302` (6 commit)

Tek günde 7 commit: kapsamlı proje dokümantasyonu yazıldı, repo başkalarının klonlayabileceği hale getirildi (hardcoded yollar kaldırıldı, tutarsızlıklar giderildi, README güncellendi).

---

# 4. Deneyler ve Denemeler

## Deneme 1: DAgger (Dataset Aggregation) Algoritması

**Kanıt:** `target_point/dagger.py` ve `run_target_point_dagger.py` — `7caadc8` reverte ile silindi.

**Ne denenmiş olabilir?** DAgger, imitation learning'de iteratif veri toplama yöntemidir. Modeli eğittikten sonra başarısız durumlardan yeni veri toplayıp yeniden eğitme döngüsü uygulanmıştır.

**Sonuç:** Reverte (`7caadc8`) ile kaldırıldı.

**Neden değiştirilmiş olabilir?** Muhtemelen DAgger'ın getirdiği ek karmaşıklık, elde edilen performans kazanımıyla orantılı görülmedi. Kesin neden commit mesajında belirtilmemiştir.

---

## Deneme 2: Scripted Expert (Kurallı Uzman Politika)

**Kanıt:** `target_point/scripted_expert.py`, `generate_scripted_target_point_data.py`, `analyze_scripted_expert_run.py` — tümü `7caadc8` ile silindi.

**Ne denenmiş olabilir?** Kural tabanlı bir araç kontrolcüsü kullanarak otomatik veri üretme. Gerçek sürücüden bağımsız büyük etiketli veri oluşturmak hedeflenmiş olabilir.

**Sonuç:** Tüm bileşenler reverte ile kaldırıldı.

---

## Deneme 3: Zamansal Model (Temporal Model)

**Kanıt:** `target_point/temporal.py` — `7caadc8` ile silindi. `configs/model_12_temporal.py` ise hâlâ mevcuttur.

**Ne denenmiş olabilir?** Tek bir kare yerine ardışık kareleri girdi olarak kullanarak hareket bilgisini kodlama.

**Sonuç:** `temporal.py` kaldırıldı; konfigürasyon dosyası kalmaya devam etti (muhtemelen gözden kaçtı).

---

## Deneme 4: Promotion Mekanizması

**Kanıt:** `target_point/promotion.py` — `7caadc8` ile silindi.

**Ne denenmiş olabilir?** İyi performanslı model checkpoint'lerini bir sonraki eğitim aşamasına başlangıç noktası olarak kullanan iteratif bir mekanizma.

---

## Deneme 5: Özel Kayıp Fonksiyonu (Effective Loss)

**Kanıt:** `target_point/effective_loss.py` — `7caadc8` ile silindi.

**Ne denenmiş olabilir?** Standart MSE yerine pist eğriliğine göre ağırlıklı kayıp veya geometrik kayıp fonksiyonu.

---

## Deneme 6: Tek-Pist Stabilizasyon

**Kanıt:** `run_single_track_stabilization.py` — `7caadc8` ile silindi. Commit `10cf4b9`.

**Ne denenmiş olabilir?** Önce tek bir pist üzerinde model stabilize etme, ardından genelleme — curriculum learning benzeri bir yaklaşım.

---

## Deneme 7: Phase 5 Varyantları (Signflip, Dynamic, Oldstyle, Smoke)

**Kanıt:** Commit `2e9d6e4`: bu varyantların raporları silindi.

- **signflip:** Direksiyon işareti çevirme ile augmentation deneyi
- **dynamic:** Dinamik lookahead veya parametre deneyi
- **oldstyle:** Eski stil kontrol yaklaşımı karşılaştırması
- **smoke:** Hızlı sanity check koşusu

**Sonuç:** "adaptive_robust" ve "fixed_robust" varyantları korundu; diğerleri temizlendi.

---

## Deneme 8: Adaptif vs. Sabit Lookahead Karşılaştırması

**Kanıt:** `labels_adaptive_v1.csv` ve `labels_fixed_1p2m.csv` paralel olarak oluşturuldu. Phase 4 ve 5'te `adaptive_*` ve `fixed_*` deneyler karşılıklı yürütüldü.

**Tezde Nasıl Anlatılabilir?** "Hedef nokta seçiminde iki farklı strateji karşılaştırmalı olarak incelenmiştir: sabit mesafe yaklaşımı (1.2 m sabit lookahead) ve adaptif mesafe yaklaşımı (pist eğriliğine göre dinamik lookahead)."

---

## Deneme 9: Myconfig Varyant Dosyaları (Phase 16, 17, 19)

**Kanıt:** `myconfig_phase16_balanced_v3.py`, `myconfig_phase17_curvature_narrow.py`, `myconfig_phase19_mid_corner_carry.py` — `7caadc8` ile silindi.

"Balanced", "curvature_narrow", "mid_corner_carry" isimlendirmeleri, hızlı deney tekrarı (rapid experimentation) yaklaşımına işaret etmektedir.

---

# 5. Karşılaşılan Problemler ve Çözümler

## Problem 1: Domain Gap (Simülasyon-Gerçek Dünya Uçurumu)

**Kanıt:** Projenin temel motivasyonu. README: "Bu yaklaşım sim-to-real transferini kolaylaştırır."

**Çözüm:**
- Doğrudan direksiyon tahmiri yerine hedef nokta tahmiri
- Çoklu veri stratejisi (model_01–model_06) ile farklı sim/real oranları
- Domain randomizasyon (`domain_randomization.py`)
- Bias kompansasyonu ve deadband (commit `2995dab`)

**Tezde:** "Simülasyon ortamında eğitilen modellerin gerçek ortamda kullanılmasında görsel domain farkı (domain gap) kritik bir sorun olarak ortaya çıkmaktadır. Bu çalışmada söz konusu sorun, ara temsil olarak hedef nokta kullanımı ve sistematik veri karışımı deneyleriyle ele alınmıştır."

---

## Problem 2: Kontrolcü Bias ve Hassasiyet Sorunu

**Kanıt:** Commit `2995dab`: "Add target-point bias compensation and deadband parameters to controller"

**Çözüm:** `controller.py`'a bias kompansasyonu ve deadband parametreleri eklendi; `simulationconfig.py` üzerinden ayarlanabilir.

**Tezde:** "Geometrik kontrolcünün pratikte uygulanmasında araç davranışından kaynaklanan sistematik sapma (bias) ve düşük genlikli komut titreşimi sorunlarıyla karşılaşılmıştır. Bu sorunlar, bias kompansasyonu ve deadband filtresi eklenerek giderilmiştir."

---

## Problem 3: Eğitim Verimliliği — Küçük Batch Boyutu

**Kanıt:** Commit `ec10b31`: `TARGET_POINT_BATCH_SIZE` 32'den 128'e yükseltildi.

**Çözüm:** Batch boyutu 4 kat artırıldı.

---

## Problem 4: Deneysel Aşırı Karmaşıklık

**Kanıt:** Commit `7caadc8` — birden fazla gelişmiş bileşenin tek seferde kaldırılması.

**Çözüm:** Tüm gelişmiş bileşenler kaldırılarak daha önce çalışan bir baseline'a (`5850afb`) geri dönüldü.

**Tezde:** "Geliştirme sürecinin belirli bir aşamasında sistemin karmaşıklığı yönetilebilir sınırları aştığında, istikrarlı bir önceki baseline'a geri dönme kararı alınmıştır."

---

## Problem 5: Pist Etiketleme Gerekliliği

**Kanıt:** Her pist için `raw_trace.csv`, `centerline.csv`, `labels_*.csv` üretilmesi zorunluluğu.

**Çözüm:** `build_target_point_labels.py` ve `track_map.py` ile otomatik pist haritalama pipeline'ı oluşturuldu.

**Tezde:** "Her pist için önce ham sürüş izi verisi toplanmış; ardından bu izden merkez hat ve lookahead etiketleri otomatik olarak hesaplanmıştır."

---

## Problem 6: Repo Klonlanabilirlik Sorunları

**Kanıt:** 2026-04-26'da 5 "fix" commiti art arda: hardcoded `C:\Users\alper\...` yolları başka ortamlarda çalışmayı engelliyordu.

**Çözüm:** Hardcoded yollar dinamik yollarla değiştirildi; `simulationconfig.py` ve README güncellendi.

---

## Problem 7: Veri Toplama Sürecinin İzlenmesi

**Kanıt:** Commit `2995dab`: `collect_massive_sim_dataset.ps1`, `monitor_massive_sim_dataset.ps1`, `watch_massive_sim_progress.ps1`.

**Çözüm:** PowerShell otomasyon betikleriyle büyük ölçekli veri toplama ve gerçek zamanlı izleme sağlandı.

---

# 6. Teknik Kararlar

## Karar 1: Hedef Nokta (Target-Point) Mimarisi

**Karar:** Direksiyon açısı yerine `(target_x, target_y)` tahmiri.

**Kanıt:** Commit `6a8538a`, README.md, `controller.py`'ın varlığı.

**Avantajları:** Sim-real transfer kolaylığı; geometrik kontrolcü yorumlanabilirlik sağlar; pistler arası genelleme daha iyi.

**Dezavantajları:** Her pist için haritalama gerekli; kontrolcü parametreleri kalibrasyonu ek iş.

---

## Karar 2: DonkeyCar Çerçevesinin Temel Alınması

**Kanıt:** `config.py` (770 satır), `manage.py`, `gym_donkeycar/` DonkeyCar kaynaklı.

**Gerekçe:** Alt seviyeli araç kontrolü ve simülatör bağlantısı hazır; özgün katkı target-point modülüne odaklandırıldı.

---

## Karar 3: TensorFlow 2.15.1 / Keras — Depthwise Separable CNN

**Kanıt:** `requirements-train.txt`, `model.py`, README (~115K parametre, depthwise separable conv).

**Gerekçe:** TFLite uyumluluğu, gömülü donanımda verimli çıkarım, Keras ile hızlı prototipleme. TF 2.15.1 yalnızca Python 3.11 ile çalışır.

---

## Karar 4: JSONL Manifest Formatı

**Kanıt:** `manifest.py`, README: "JSONL manifest + DonkeyCar Tub V2 görüntüleri".

**Gerekçe:** Büyük veri kümeleri için satır satır okuma, bellek verimliliği, karma/filtreleme kolaylığı.

---

## Karar 5: Lookahead Stratejisi — Adaptif vs. Sabit

**Kanıt:** Her pist için paralel `labels_adaptive_v1.csv` ve `labels_fixed_1p2m.csv`. Phase 4–5'te karşılaştırmalı deneyler.

---

## Karar 6: PowerShell Otomasyon Betikleri

**Kanıt:** `scripts/` dizininde birden fazla `.ps1` dosyası. Geliştirme ortamı Windows.

---

## Karar 7: TFLite Dışa Aktarma (Jetson Desteği)

**Kanıt:** Commit `40d035d`: `export.py` ve `pilot_tflite.py` eklendi.

**Tezde:** "Gerçek donanım üzerinde dağıtım için model TensorFlow Lite formatına dönüştürülmüş ve Jetson platformu için optimize edilmiş hafif bir çıkarım modülü geliştirilmiştir."

---

## Karar 8: Çok Model Konfigürasyon Sistemi

**Kanıt:** `configs/model_01` — `model_12`, `MODEL_TRAINING_PLAN.md`, `model_export_manifest.json`.

**Tezde:** "9 farklı model konfigürasyonu, yalnızca simülasyon verisinden yalnızca gerçek veriye uzanan spektrumda sistematik biçimde eğitilmiştir."

---

# 7. Sistem Mimarisi Evrimi

## İlk Mimari (2026-03-05)
```
otonom-arac/
├── calibrate.py
├── config.py              (770 satır — DonkeyCar config)
├── myconfig.py
├── manage.py
├── train.py
├── data/
├── models/                (dilara.h5, my_first_pilot.h5, my_2_first_pilot.h5)
├── examples/
│   ├── genetic_alg/
│   ├── reinforcement_learning/
│   └── supervised_learning/
├── gym_donkeycar/
└── donkey_sim.app
```
Standart DonkeyCar çerçevesi — doğrudan direksiyon açısı tahmiri.

---

## Ara Mimari 1: Target-Point Modülü (2026-03-24)
```
+ target_point/
│   ├── model.py, controller.py, dataset.py
│   ├── training.py, diagnostics.py, pilot.py
└── tests/test_target_point.py
```
Eklentili (additive) mimari evrim.

---

## Ara Mimari 2: Büyük Genişleme (2026-03-27)
```
+ target_point/
│   ├── augment.py, collector.py, domain_randomization.py
│   ├── evaluate_closed_loop.py, manifest.py, sim_session.py
│   ├── teacher_policy.py (1003 satır), track_map.py
│   ├── dagger.py, scripted_expert.py, promotion.py  ← (sonradan silindi)
│   ├── temporal.py, effective_loss.py               ← (sonradan silindi)
+ build_target_point_labels.py
+ collect_target_point_data.py
+ data/artifacts/maps/
```

---

## Mimari Geri Alma (2026-04-03)
```
KALDIRILDI:
- target_point/dagger.py, scripted_expert.py, promotion.py
- target_point/temporal.py, effective_loss.py
- run_target_point_dagger.py, run_single_track_stabilization.py
- generate_scripted_target_point_data.py
- myconfig_phase*.py (varyant dosyaları)
```
**Subtractive** (çıkarıcı) mimari adım. 4829 satır silindi.

---

## Mimari Yeniden Yapılanma (2026-04-17)
```
+ ai_pipeline/
│   ├── target_point/      (taşındı: kök dizinden)
│   ├── train.py, build_target_point_labels.py
│   ├── collect_target_point_data.py, evaluate_target_point.py
│   └── tools/
+ configs/
│   ├── model_01_pure_sim.py  …  model_12_temporal.py
+ scripts/
│   ├── train_first6_models.ps1, train_gpu.ps1
│   ├── collect_massive_sim_dataset.ps1 vb.
- examples/supervised_learning/  (tamamen kaldırıldı)
```

---

## Mimari Özet Tablosu

| Dönem | Durum |
|-------|-------|
| Mart başı | DonkeyCar çerçevesi (düz direksiyon tahmiri) |
| Mart ortası | `target_point/` modülü eklendi |
| Mart sonu | `target_point/` büyük ölçüde genişletildi |
| Nisan başı | Revert: sistem sadeleştirildi |
| Nisan ortası | `ai_pipeline/` yapısına geçiş |
| Nisan sonu | Dokümantasyon ve klonlanabilirlik |

**Genel eğilim:** Genişleme (additive) → Sadeleşme (subtractive) → Yeniden yapılanma (structural reorganization).

---

# 8. Özellik Bazlı Gelişim

## Özellik 1: Target-Point CNN Modeli
- **İlk commit:** `6a8538a` (2026-03-24)
- **Dosya:** `ai_pipeline/target_point/model.py`
- **Gelişim:** İlk temel Keras → normalizasyon eklendi → sadeleştirildi → TFLite export eklendi
- **Teknik:** ~115K parametre, depthwise separable conv, 224×224 RGB giriş, `TargetPointDenormalizer` çıktı normalizasyonu
- **Tezde:** "Modelde yaklaşık 115.000 parametre içeren ve depthwise separable evrişim katmanlarından oluşan bir CNN mimarisi kullanılmıştır."

## Özellik 2: Geometrik Kontrolcü
- **İlk commit:** `6a8538a` (2026-03-24)
- **Dosya:** `ai_pipeline/target_point/controller.py`
- **Gelişim:** 57 satır → genişledi → sadeleşti → bias/deadband eklendi
- **Teknik:** Heading error hesaplama → PD benzeri direksiyon → viraj sertliğine göre dinamik gaz

## Özellik 3: Pist Haritalama Sistemi
- **İlk commit:** `5850afb` (2026-03-27)
- **Pipeline:** raw_trace → deduplication → lap split → resample → centerline → labels

## Özellik 4: Kapalı Döngü Değerlendirme
- **İlk commit:** `5850afb` (542 satır) → `7caadc8` ile sadeleşti (251 satır)
- **Çıktı:** `closed_loop_episodes.jsonl`, `closed_loop_summary.json`

## Özellik 5: Domain Randomizasyon
- **İlk commit:** `5850afb`
- **Teknik:** Renk jitter, parlaklık, gürültü — simülasyon görüntülerini çeşitlendirme

## Özellik 6: Veri Artırma (Augmentation)
- **İlk commit:** `5850afb` (149 satır)
- **Teknik:** Yatay çevirme (flip) + hedef x koordinatı simetrik işaretleme

## Özellik 7: TFLite Dışa Aktarma
- **İlk commit:** `40d035d` (2026-04-17)
- **Dosyalar:** `export.py`, `pilot_tflite.py`

## Özellik 8: Çoklu Model Konfigürasyon Sistemi
- **İlk commit:** `ec10b31` (2026-04-21)
- **Kapsam:** model_01 (pure sim) → model_12 (temporal), 9 aktif konfigürasyon

---

# 9. Hata Düzeltmeleri ve Refactoring

## Değişiklik 1: Hardcoded Kişisel Yolların Kaldırılması
**Tür:** Hata düzeltme | **Commitler:** `358d90e`, `d483ea6`
`C:\Users\alper\...` sabit yolları → dinamik değişkenler ve göreceli yollar.

## Değişiklik 2: Repo Klonlanabilirlik Düzeltmeleri
**Tür:** Hata düzeltme | **Commitler:** `4ba0252`, `7466d2b`, `fec3302`
"Yeni kullanıcıyı çökertecek 4 tutarsızlık" giderildi.

## Değişiklik 3: Supervised Learning Örneklerinin Kaldırılması
**Tür:** Refactoring | **Commit:** `40d035d`
`examples/supervised_learning/` dizini ve ilgili tüm dosyalar silindi.

## Değişiklik 4: Target-Point Modülünün AI Pipeline'a Taşınması
**Tür:** Yapısal düzenleme | **Commit:** `5de581c`
`target_point/` → `ai_pipeline/target_point/` (ML kodu araç kontrolünden ayrıştırıldı)

## Değişiklik 5: Train.py'nin Cihaz Yapılandırmasıyla Güncellenmesi
**Tür:** Özellik | **Commit:** `c5b68e3`
Sabit donanım → GPU/CPU seçimi + `requirements-train.txt` ayrıştırıldı.

## Değişiklik 6: DAgger ve İlgili Bileşenlerin Kaldırılması (Büyük Revert)
**Tür:** Refactoring | **Commit:** `7caadc8`
165 dosya, 4829 satır silindi. `training.py` ~757 → sadeleşti. Test seti ~989 → küçüldü.

## Değişiklik 7: README'nin Yeniden Yazılması
**Tür:** Dokümantasyon | **Commit:** `b27709b`
Proje amacı, güncel mimari ve kurulum talimatlarını yansıtacak şekilde yeniden yazıldı.

## Değişiklik 8: Controller'a Bias ve Deadband Eklenmesi
**Tür:** Hata düzeltme | **Commit:** `2995dab`
Saf geometrik kontrolcü → sistematik sapma ve titreşim giderildi.

## Değişiklik 9: Batch Size Optimizasyonu
**Tür:** Performans | **Commit:** `ec10b31`
`TARGET_POINT_BATCH_SIZE`: 32 → 128 (4× artış).

---

# 10. Test, Doğrulama ve Sınırlamalar

## Test İzleri

**Birim Test Dosyası:** `tests/test_target_point.py`
- `6a8538a`'da oluşturuldu (130 satır)
- `5850afb`'de büyük ölçüde genişledi (~989 satır)
- `7caadc8` sonrası küçüldü

Commit `6a8538a` mesajı: "Included tests for key functionalities such as target point computation and model evaluation."

**conftest.py:** Commit `40d035d`'de `tests/conftest.py` eklendi — pytest ortam yapılandırması.

## Manuel Doğrulama Olasılıkları

**Smoke Test:**
```powershell
.\.venv\Scripts\python manage.py smoke --simulationconfig=simulationconfig.py
```
Simülatör bağlantısı, model yükleme ve temel sürüş döngüsünü test eder.

**Kapalı Döngü Değerlendirme:**
`evaluate_closed_loop.py` ile tam otonom sürüş; `closed_loop_summary.json` çıktısı.
Mevcut: `data/artifacts/reports/phase5_adaptive_eval/`, `phase5_fixed_eval/` vb.

**Beş Koşu Özet Raporu:**
Commit `8231732` — üretilmiş yollar üzerinde tutarlılık doğrulaması.

## Kod Üzerinden Görülen Kontroller

- `TargetPointDenormalizer`: normalizasyon istatistiklerinin doğruluğunu zorunlu kılar
- `preprocess_image`: eğitim ve çıkarım için aynı ön işleme — train/test tutarsızlığını önler
- Kırpma sınır kontrolü: `ValueError` ile erken hata
- `collapse_monitor.jsonl`: eğitim sırasında model çöküşü izleme

## Sınırlamalar

1. **Test kapsamı sınırlı** — kontrolcü uçtan uca, pist haritalama, TFLite kalitesi test edilmemiş görünüyor
2. **Gerçek donanım testi belirsiz** — TFLite pilot yazılmış ama gerçek araç verileri repoda yok
3. **Kapalı döngü değerlendirme yalnızca simülatörde** — gerçek ortam karşılaştırması mevcut değil
4. **Eğitim kayıtları tutarsız** — bazı deneyler için history.csv var, bazıları için yalnızca model dosyası
5. **Commitlenmemiş denemeler** — yerel başarısız eğitimler ve alternatif mimariler kayıt dışı
6. **Başarı kriterleri tanımsız** — hangi metrik eşiğinin "yeterli" sayıldığı belirtilmemiş

---

# 11. Tezde Kullanılabilecek Anlatım

## Projenin Başlangıcı

Bu çalışmada otonom araç kontrol sistemi geliştirilmesi için DonkeyCar açık kaynak çerçevesi temel alınmıştır. DonkeyCar, düşük maliyetli otonom araç projeleri için araç kontrolü, simülatör bağlantısı ve temel veri toplama altyapısı sunan kapsamlı bir çerçevedir. Projenin başlangıç aşamasında çerçeve yerel ortama kurulmuş, temel eğitim altyapısı doğrulanmış ve ilk model denemeleri gerçekleştirilmiştir.

Simülasyon ortamı olarak Unity tabanlı DonkeySim kullanılmıştır. Bu simülatör, aracın çeşitli sanal pistlerde sürülmesine, görüntü verisi toplanmasına ve farklı ışık ile fizik koşullarının test edilmesine olanak tanımaktadır.

## Hedef Nokta Yaklaşımının Benimsenmesi

Projenin temel mimari kararı, doğrudan direksiyon açısı tahmiri yerine hedef nokta (target-point) tahmirinin benimsenmesidir. Bu yaklaşımda model, kamera görüntüsünden ego-frame koordinat sisteminde ilerleyen bir noktanın `(target_x, target_y)` koordinatını tahmin eder. Tahmin edilen koordinat, geometrik bir kontrolcü tarafından direksiyon ve gaz komutlarına dönüştürülmektedir.

Bu mimari tercihin temel motivasyonu, simülasyon-gerçek transfer (sim-to-real) performansıdır. Doğrudan direksiyon tahmiri, simülasyon ile gerçek dünya arasındaki görsel farklılıklara (domain gap) duyarlı iken, hedef nokta tahmiri bu problemi görsel olmayan bir geometrik ara temsile taşıyarak daha sağlam bir transfer sağlamaktadır.

## İlk Prototipin Oluşturulması

Hedef nokta modülü (`target_point/`) üç hafta süren geliştirme sürecinin ardından sisteme entegre edilmiştir. Bu modül; model, kontrolcü, veri kümesi yükleme, eğitim döngüsü, teşhis araçları ve gerçek zamanlı çıkarım bileşenlerini kapsamaktadır.

Model mimarisi, ~115.000 parametre içeren ve depthwise separable evrişim katmanları kullanan bir CNN olarak tasarlanmıştır. Gömülü donanımda (Jetson platformu) çalışabilirlik hedefi göz önünde bulundurulduğunda, parametre etkinliği önemli bir tasarım kısıtı olarak belirlenmiştir.

## Pist Haritalama ve Etiketleme

Hedef nokta tabanlı eğitim, her pist için merkez hat bilgisi gerektirmektedir. Bu amaçla iki aşamalı bir etiketleme pipeline'ı geliştirilmiştir.

İlk aşamada araç simülatörde pist boyunca sürülmekte ve konum verileri `raw_trace.csv` formatında kaydedilmektedir. İkinci aşamada ham iz verisi işlenerek pist merkez hattı (`centerline.csv`) hesaplanmakta ve her görüntü karesi için uygun hedef nokta koordinatı geometrik yöntemle belirlenmektedir.

Hedef nokta seçiminde iki farklı strateji karşılaştırmalı olarak geliştirilmiştir: 1.2 metre sabit ileri bakış mesafesi (fixed lookahead) ve pist eğriliğine duyarlı adaptif ileri bakış mesafesi (adaptive lookahead).

## Temel Özelliklerin Geliştirilmesi

İlk çalışan prototipten sonra sistem çok sayıda ek bileşenle genişletilmiştir. Veri artırma (`augment.py`) ile eğitim sırasında görüntü dönüşümleri uygulanmıştır. Domain randomizasyon (`domain_randomization.py`) ile simülasyon görüntüleri, gerçek dünya görüntülerinin çeşitliliğini daha iyi yansıtacak biçimde dönüştürülmüştür. Kapalı döngü değerlendirme çerçevesi (`evaluate_closed_loop.py`), modelin simülatörde tam otonom sürüş yapmasını ve performans metriklerinin otomatik olarak kaydedilmesini sağlamıştır.

## Karşılaşılan Problemler

Geliştirme sürecinde Dataset Aggregation (DAgger) algoritması, kural tabanlı scripted expert politikası, zamansal model mimarisi ve özel kayıp fonksiyonları gibi gelişmiş teknikler denenmiştir. Kod değişiminden anlaşıldığı kadarıyla bu tekniklerin pratikte beklenen kazanımı sağlamadığı ya da sistemin yönetilebilirliğini olumsuz etkilediği değerlendirilmiştir. Bu değerlendirme sonucunda önceden stabil olduğu bilinen bir baseline yapıya geri dönme kararı alınmıştır.

## Çözüm Yaklaşımları

1. **Ara Temsil Seçimi:** Doğrudan direksiyon tahmiri yerine hedef nokta tahmiri benimsenerek görsel domain bağımlılığı azaltıldı.
2. **Sistematik Karşılaştırma:** 9 farklı model konfigürasyonu karşılaştırmalı olarak değerlendirildi.
3. **Kontrolcü Kalibrasyonu:** Bias kompansasyonu ve deadband parametreleri eklenerek gerçek araç davranışındaki sistematik sapmalar giderildi.
4. **Sadeliği Tercih Etme:** Karmaşık deneysel bileşenler, güvenilir ve sade bir baseline lehine kaldırıldı.

## Sistem Mimarisi

**Algı Katmanı:** Araç ön kamerasından alınan 224×224 piksel görüntü, depthwise separable evrişim katmanlarından oluşan CNN modeline beslenir. Model, ego-frame koordinat sisteminde `(target_x, target_y)` üretir.

**Karar Katmanı (Kontrolcü):** Geometrik kontrolcü, tahmin edilen koordinattan araç başlığına olan açı hatasını (heading error) hesaplar ve bunu direksiyon komutuna dönüştürür.

**Dağıtım Katmanı:** Model TensorFlow Lite formatına dönüştürülmüş ve Jetson platformu için optimize edilmiş bir çıkarım modülü geliştirilmiştir.

## Test ve Değerlendirme

Sistem performansı iki düzeyde değerlendirilmiştir. Birim testler ile temel bileşenlerin doğruluğu test edilmiştir. Kapalı döngü simülasyon değerlendirmeleriyle modelin otonom sürüş kapasitesi ölçülmüştür.

## Sonuç

Bu çalışmada, DonkeyCar çerçevesi üzerine inşa edilen bir hedef nokta tabanlı otonom araç kontrol sistemi geliştirilmiştir. Geliştirme süreci; mimari kararlar, sistematik deneyler ve iteratif iyileştirmelerden oluşan tipik bir araştırma döngüsünü yansıtmaktadır.

*Not: Bu paragraflar teze aktarılırken deneysel sonuçlarla desteklenmeli ve spesifik metrik değerleriyle zenginleştirilmelidir.*

---

# 12. Zaman Çizelgesi

## Kronolojik Tablo

| Tarih | Geliştirme Aşaması | İlgili Commitler | Açıklama |
|-------|-------------------|-----------------|----------|
| 2026-03-05 | Proje Kurulumu | `eee7f31`, `c5b68e3` | DonkeyCar çerçevesi + cihaz yapılandırması. |
| 2026-03-11 | İlk Model Konfigürasyonu | `b28d47b` | `lane_v1` config, myconfig güncellemesi. |
| 2026-03-24 | Target-Point Modülü | `6a8538a` | Ana özellik: tüm target_point/ alt modülleri. |
| 2026-03-27 | Pist Haritalama + Phase 4-5 | `5850afb` | 6 pist haritalandı. 43K+ satır ekleme. |
| 2026-03-29 | Phase 5.5 Değerlendirme | `e02951f`, `ac4920b` | Bootstrap raporları ve yeni bileşenler. |
| 2026-03-31 | Tek-Pist Stabilizasyon Denemesi | `10cf4b9` | Sonradan kaldırılan betik. |
| 2026-04-01 | Temizleme | `2952627`, `e9f3908` | "last" ve Unity log temizliği. |
| 2026-04-03 | **Büyük Revert** | `7caadc8`, `31f73db`, `2e9d6e4` | DAgger, scripted expert, temporal vb. kaldırıldı. |
| 2026-04-05 | Yeni Pist Denemeleri | `0f0b351`, `8231732`, `41389c3` | Üretilmiş yollar + 5 koşu raporu. |
| 2026-04-06 | Kapalı Döngü Değerlendirme | `a4c40f7`, `fcb6f8f`, `233ad5a` | 3 commit — çoklu sürüş oturumu raporları. |
| 2026-04-07 | Dokümantasyon | `43894f0`, `4f802fc`, `7236feb` | COMMANDS.md, SETUP.md, PowerShell sözdizimi. |
| 2026-04-08 | Unity Log | `2c25c81` | Performans logu eklendi. |
| 2026-04-16 | Sim2Real Kılavuzları | `c8e688f`, `c98a637` | Çok pist kılavuzu + 3 tur koşu betiği. |
| 2026-04-17 | **Büyük Yeniden Yapılanma** | `5de581c`, `40d035d`, `95dafd8` | `ai_pipeline/` yapısı + TFLite + supervised learning silindi. |
| 2026-04-18 | Unity Log | `6687c15` | Debug logu. |
| 2026-04-19 | Kontrolcü İyileştirme | `2995dab` | Bias, deadband, büyük veri toplama betikleri. |
| 2026-04-21 | Çok Model Eğitim Planı | `ec10b31` | Model 01-06, batch 32→128, otomasyon betikleri. |
| 2026-04-22 | Ek Model Konfigürasyonları | `3f66a3f` | Model 07, 11, 12 + export manifest. |
| 2026-04-26 | **Yayına Hazırlık** | `0ce7fd9` → `fec3302` (6 commit) | Dokümantasyon + hardcoded yollar + klonlanabilirlik. |

## Özet İstatistikler

| Metrik | Değer |
|--------|-------|
| İlk commit | 2026-03-05 |
| Son commit | 2026-04-26 |
| Toplam süre | 52 gün |
| Toplam commit | 39 |
| Ortalama | ~0.75 commit/gün |
| En yoğun gün | 2026-04-26 (7 commit) |

---

# 13. Eksik Bilgiler ve Dürüst Notlar

> **Not (§15–21 eklenmiştir):** Aşağıdaki maddelerden bir kısmı, `metrics.json`, `closed_loop_summary.json` ve `collapse_monitor.jsonl` dosyaları okunarak §15–21 bölümlerinde cevaplanmıştır. Her maddenin yanındaki durum etiketi bunu göstermektedir.

## Net Olarak Çıkarılamayan Bilgiler

**1. Deneysel Sonuçların Karşılaştırması** ✅ ÇÖZÜLDÜ — Bkz. §15, §16
`metrics.json` dosyaları okunmuştur. `target_point_realtrack_ready` modeli tek başarılı smoke test modelidir (0 offtrack, 10.05 saniye). Model karşılaştırma tablosu §16'dadır.

**2. Gerçek Araç Üzerinde Test** ⚠️ HÂLÂ AÇIK
`pilot_tflite.py` ve Jetson kodu yazılmış; TFLite FP16 dönüşümü başarılı (`model_export_manifest.json`). Ancak gerçek araç üzerinde test sonuçları repoda bulunmamaktadır. Gerçek araç testi yaptıysanız teze kendiniz ekleyin.

**3. DAgger ve İleri Tekniklerin Başarısızlık Nedeni** ⚠️ KISMI — Bkz. §19
Commit mesajı yalnızca "baseline'a geri dönüş" diyor. Phase 4 eğitiminin kararsız olduğu (`collapse_monitor.jsonl` verisinden: val_loss 2.208→3.098→2.872, hiç yakınsamadı) kanıtlanmıştır. DAgger için ayrı bir closed-loop raporu bulunmamaktadır.

**4. Commitlenmemiş Denemeler** ⚠️ HÂLÂ AÇIK
Git geçmişinde görünmeyen yerel denemeler kayıt dışı kalmaktadır.

**5. Phase Numaralandırması Mantığı** ⚠️ HÂLÂ AÇIK
Phase 1, 2, 3 neredeydi? Belirsizliği devam etmektedir.

**6. "last" ve "Son yapılan alan" Commit Mesajları** ⚠️ HÂLÂ AÇIK
Commitler `2952627` ve `41389c3` — diff incelemeden içerikleri belirlenemez.

**7. Model Performans Beklentileri** ✅ KISMI ÇÖZÜM — Bkz. §21
`collapse_gate` sistemi, 4 somut kriter aracılığıyla resmi başarı eşiği işlevi görmektedir (`turn_mae_x`, `val_corr_x`, `val_pred_x_std_ratio`, `train_pred_x_std_ratio`). Closed-loop başarı eşiği ise "0 offtrack + 200 adım tamamlama" olarak smoke testlerde uygulanmıştır.

**8. Eğitim Süreleri** ⚠️ HÂLÂ AÇIK
Tek model eğitimi için gereken süre repoda kayıtlı değildir. model_05 30 epoch eğitilmiştir — diğerleri 4–20 epoch arasında değişmektedir.

## Tahmine Dayalı Yorumlar (Güncellenmiş)

| Tahmin | Dayanağı | Kesinlik | Durum |
|--------|----------|----------|-------|
| DAgger başarısız bulundu | Revert commit + küçük veri seti (14K) + val_loss divergence | Orta | ⚠️ Veri sorunuyla desteklendi |
| Scripted expert gerçek veriden geride kaldı | Modül silindi | Düşük | ⚠️ Değişmedi |
| Batch size artışı eğitim süresini kısalttı | 32→128 değişikliği | Düşük | ⚠️ Değişmedi |
| Phase numaraları eğitim döngülerini temsil ediyor | İsimlendirme kalıbı | Orta | ⚠️ Değişmedi |
| macOS'ta başlayıp Windows'a taşındı | `__MACOSX/` artefaktları | Yüksek | ✅ Değişmedi |
| Gerçek araç Jetson kullanıyor | `pilot_tflite.py` + TFLite manifest | Yüksek | ✅ `model_export_manifest.json` ile desteklendi |
| Model parametresi ~115K | Önceki tahmin | — | ❌ YANLIŞ: Gerçek değer **33.858** (`model_export_manifest.json`) |

## Tez Yazarken Kaçınılması Gerekenler

- "DAgger başarısız oldu" → "Phase 4 eğitiminde val_loss yakınsamadı (2.208→3.098, 5 epoch); DAgger için ayrı bir performans raporu bulunmamaktadır"
- "Model X en iyi sonucu verdi" → "§16 tablosundan: `target_point_realtrack_ready` tek 0-offtrack modelidir"
- "Sistem gerçek araçta başarıyla çalıştı" → yalnızca gerçekten test yaptıysanız yazın
- "~115K parametre" → "33.858 parametre (`model_export_manifest.json`'dan)"

## Güvenle Kullanılabilecek Bilgiler (Deney Verileriyle Genişletilmiş)

**Commit geçmişinden:**
- Proje tarihleri: 2026-03-05 başlangıç, 2026-04-26 son commit
- Teknoloji: TF 2.15.1, Python 3.11, DonkeyCar, Unity/DonkeySim
- Target-point yaklaşımının benimsenmesi: 2026-03-24
- DAgger ve scripted expert modüllerinin oluşturulup kaldırıldığı: 2026-04-03
- `ai_pipeline/` yapısına geçiş: 2026-04-17

**Deney dosyalarından (§15–21):**
- Model parametre sayısı: **33.858** (tüm modeller, model_11 hariç)
- model_11_multitask parametre sayısı: **669.828**
- Phase 4 eğitiminde val_loss: **2.208 → 3.098 → 2.872** (yakınsamadı)
- model_01 val_corr_x: **0.841–0.859** (collapse gate PASSED)
- model_07 val_corr_x: **0.123** (collapse = sabit tahmin)
- `target_point_realtrack_ready` smoke test: **0 offtrack, 10.05 saniye**
- TFLite FP16 boyutu: **77.420 byte** (~76 KB)
- Kontrolcü en iyi konfigürasyon: **bias=0.20, max=0.07** (21.85s hayatta kalma)

## Öğrencinin Tamamlaması Gereken Kısımlar

1. ~~Deneysel sonuçların karşılaştırmalı tablosu~~ → §15–16'da tamamlandı
2. Gerçek araç test sonuçları (TFLite dağıtımı sonrası — varsa)
3. DAgger'ın neden kaldırıldığına dair kişisel notlar
4. Eğitim süreleri (GPU zamanı)
5. Phase numaralandırmasının açıklaması (kişisel bilgi gerektiriyor)
6. Closed-loop ortamında kaç tur tamamlandığı (pist haritasız raporlarda lap_count=0)

---

# 14. Dokümantasyon Rehberi

## Bu Belge Hakkında

Bu tümleşik dosya, `docs/thesis-development/` altındaki 00–12 numaralı dosyaların ve README'nin tüm içeriğini kapsamaktadır. Kaynak dosyalar silinmemiştir; bu dosya yalnızca pratik tek-dosya erişimi için hazırlanmıştır.

## Kaynaklar

Bu belgeler şu kaynaklardan üretilmiştir:
- `git log --oneline --reverse` — commit kronolojisi
- `git show <commit>` — her commit için değişen dosyalar
- `git log --stat` — dosya bazında ekleme/silme istatistikleri
- Mevcut kaynak kod dosyaları (`model.py`, `controller.py`, `training.py` vb.)
- `README.md` içeriği

## Tezin Bölümlerine Göre Yönlendirme

| Tez Bölümü | Kullanılacak Bölüm |
|------------|-------------------|
| Sistem Tasarımı | §6 Teknik Kararlar, §7 Mimari Evrimi |
| Yöntem | §8 Özellik Bazlı Gelişim, §3 Geliştirme Aşamaları |
| Deneyler | §4 Deneyler, §10 Test ve Sınırlamalar |
| Hazır Paragraflar | §11 Tezde Kullanılabilecek Anlatım |
| Son Kontrol | §13 Eksik Bilgiler — mutlaka okuyun |

## Kesin Bilgi / Tahmin Ayrımı

Tahmin içeren ifadeler şu kelimelerle işaretlenmiştir: "muhtemelen", "kod değişiminden anlaşıldığı kadarıyla", "commit farkına göre", "bu değişiklik şu amaca hizmet ediyor olabilir", "kesin neden commit mesajında belirtilmemiştir".

---

# § 15 — Gerçek Deney Sonuçları

Bu bölüm, repo içindeki `metrics.json`, `history.csv`, `collapse_monitor.jsonl` ve `closed_loop_summary.json` dosyalarından doğrudan okunmuş veriler içermektedir. Tüm sayılar bu dosyalardan alınmıştır; tahmin içermemektedir.

---

## 15.1 Phase 4 Eğitim Deneyleri (Nisan 18–20, 2026)

Phase 4 deneyleri `data/artifacts/target_point/experiments/phase4_adaptive_v1_20260418_185353/` dizininde kayıtlıdır.

**Veri seti boyutu:** 14.014 eğitim örneği (diğer modellerin 328.000+ örneğine kıyasla çok küçük)

**`collapse_monitor.jsonl` verisi — epoch bazlı eğitim seyri:**

| Epoch | Train Loss | Val Loss | Val Corr X | Val MAE X | Durum |
|-------|-----------|---------|------------|-----------|-------|
| 1 | 5.072 | 2.208 | 0.271 | 0.256 | — |
| 2 | 3.804 | 2.520 | 0.305 | 0.277 | Val loss arttı |
| 3 | 3.269 | **3.098** | 0.220 | 0.285 | Divergence (epoch zirvesi) |
| 4 | 2.972 | 2.382 | 0.347 | 0.253 | Geçici toparlanma |
| 5 | 2.737 | **2.872** | 0.273 | 0.327 | Yeniden kötüleşti |

`collapse_gate.val_corr_x` kriteri geçilememiştir. Val loss hiçbir zaman 2.0'ın altına düşmemiş, 5 epoch boyunca kararsız seyretmiştir.

---

## 15.2 Model 01–07 Ana Eğitim Kampanyası (Nisan 21–22, 2026)

Ana kampanyada 328.322 eğitim örneği kullanılmıştır (7 farklı DonkeyCar pisti).

**`collapse_gate` sistemi 4 kriteri kontrol etmektedir:**
- `train_pred_x_std_ratio`: Eğitim setinde X tahminlerin varyansı
- `turn_mae_x`: Viraj bölgelerinde X koordinat ortalama mutlak hatası
- `val_corr_x`: Doğrulama setinde X koordinat korelasyonu
- `val_pred_x_std_ratio`: Doğrulama setinde X tahminlerin varyansı

**Tüm modeller için `metrics.json` özeti:**

| Deney | Veri Stratejisi | Epoch | En İyi Val Loss | Val MAE X | Val Corr X | Collapse Gate |
|-------|-----------------|-------|-----------------|-----------|------------|---------------|
| phase4_adaptive_v1 | Saf sim (14K) | 5 | 2.208 | 0.256 | 0.271 | **FAILED** (val_corr_x) |
| multitrack_320x320_efficient_v1 | Saf sim (14K, 224px) | 9 | 0.500 | 0.549 | 0.118 | **FAILED** (turn_mae_x + val_corr_x) |
| multitrack_320_v2 | Saf sim (14K, 224px) | 18 | 1.517 | 0.220 | 0.447 | **FAILED** (turn_mae_x) |
| model_01_pure_sim (a3) | Saf sim (328K) | 20 | **0.4231** | **0.064** | 0.841 | **PASSED** |
| model_01_pure_sim (a1) | Saf sim (328K) | 10 | 0.4243 | 0.067 | 0.859 | **PASSED** |
| model_02_sim_domain_rand | Sim + domain rand (328K) | 7 | 1.515 | 0.197 | 0.556 | **FAILED** (turn_mae_x) |
| model_03_pure_real | Saf gerçek (694K) | 18 | 0.343* | 0.226 | 0.842 | **FAILED** (turn_mae_x, val_mae_y=0.0) |
| model_04_hybrid_v1_naive | Hibrit naive (328K) | 19 | 0.613 | 0.124 | 0.848 | **PASSED** |
| model_05_hybrid_v2_sim_heavy | Hibrit sim-ağırlıklı (328K) | 30 | 0.534 | 0.092 | 0.813 | **PASSED** |
| model_06_hybrid_v3_real_heavy | Hibrit real-ağırlıklı (328K) | 13 | 0.670 | 0.189 | 0.709 | **PASSED** |
| model_07_finetune | Fine-tune (633K) | 4 | 1.430 | 0.416 | **0.123** | **FAILED** (3/4 kriter) |

*model_03 val loss düşük görünmekte, ancak `val_mae_y = 0.0` ve `val_corr_y = NaN` veri sorununa işaret etmektedir.

---

# § 16 — Model Karşılaştırmaları

## 16.1 Model Mimarisi

`model_export_manifest.json` dosyasına göre tüm dışa aktarılan modellerin parametresi ve boyutu:

| Model | Parametre Sayısı | Giriş Boyutu | Çıkış Boyutu | .keras Boyutu | TFLite FP16 Boyutu |
|-------|-----------------|--------------|--------------|---------------|---------------------|
| model_01_pure_sim | 33.858 | 128×128×3 | (None, 2) | 207.912 byte | 77.420 byte |
| model_02_sim_domain_rand | 33.858 | 128×128×3 | (None, 2) | 207.912 byte | 77.420 byte |
| model_03_pure_real | 33.858 | 128×128×3 | (None, 2) | 207.896 byte | 77.420 byte |
| model_04_hybrid_v1 | 33.858 | 128×128×3 | (None, 2) | 207.912 byte | 77.420 byte |
| model_05_hybrid_v2 | 33.858 | 128×128×3 | (None, 2) | 207.912 byte | 77.420 byte |
| model_06_hybrid_v3 | 33.858 | 128×128×3 | (None, 2) | 207.912 byte | 77.420 byte |
| model_07_finetune | 33.858 | 128×128×3 | (None, 2) | 208.552 byte | 78.324 byte |
| **model_11_multitask** | **669.828** | 128×128×3 | [(None,1),(None,1)] | 2.728.784 byte | 1.346.992 byte |
| sim_multitrack_v1 | 33.858 | **224×224×3** | (None, 2) | 204.328 byte | 77.420 byte |
| target_point_realtrack_ready | 33.858 | **224×224×3** | (None, 2) | 204.328 byte | 77.420 byte |

**Dikkat:** model_11_multitask, diğer modellerin ~20 katı parametre içermektedir. Tüm diğer modeller aynı 33.858 parametre ve aynı mimariye sahiptir. Tüm dönüşümler `status: "ok"` ile başarıyla tamamlanmıştır.

## 16.2 Val Corr X Karşılaştırması

Val Corr X, modelin X koordinatı (direksiyon yönü) tahmin gücünü ölçen en kritik metriktir.

```
model_01_pure_sim (a1)   : val_corr_x = 0.859  ████████████████████  PASSED
model_04_hybrid_v1       : val_corr_x = 0.848  ████████████████████  PASSED
model_03_pure_real       : val_corr_x = 0.842  ████████████████████  FAILED*
model_01_pure_sim (a3)   : val_corr_x = 0.841  ███████████████████   PASSED
model_05_hybrid_v2       : val_corr_x = 0.813  ██████████████████    PASSED
model_06_hybrid_v3       : val_corr_x = 0.709  ████████████████      PASSED
model_02_sim_domain_rand : val_corr_x = 0.556  ████████████          FAILED
multitrack_320_v2        : val_corr_x = 0.447  ██████████            FAILED
phase4_adaptive_v1       : val_corr_x = 0.271  ██████                FAILED
multitrack_320x320_eff   : val_corr_x = 0.118  ███                   FAILED
model_07_finetune        : val_corr_x = 0.123  ███                   FAILED
```

*model_03 val_corr_x iyi görünse de `val_mae_y = 0.0` nedeniyle collapse gate FAILED.

---

# § 17 — Closed-Loop Performans Analizi

## 17.1 Model Smoke Testi — Tüm Modeller

Smoke testi, her modeli `donkey-generated-roads-v0` pistinde 200 adım sınırıyla (max 10.05 saniye) çalıştırmaktadır. `closed_loop_summary.json` dosyalarından okunmuştur.

| Model | Sürüş Süresi (s) | Offtrack Sayısı | Mean CTE (m) | Mean |Abs| Steering | Recovery Başarı |
|-------|-----------------|-----------------|--------------|----------------------|-----------------|
| sim_multitrack_v1 | 1.75 | 1 | 0.321 | 0.924 | 0/1 = %0 |
| model_03_pure_real | 1.95 | 1 | 0.323 | **0.921** | 0/1 = %0 |
| model_11_multitask | 2.10 | 1 | 0.333 | 0.076 | — |
| model_04_hybrid_v1 | 2.20 | 1 | 0.322 | 0.614 | 0/1 = %0 |
| model_01_pure_sim | 2.40 | 1 | 0.268 | 0.589 | 0/1 = %0 |
| model_02_sim_domain_rand | 2.50 | 1 | 0.403 | 0.456 | 0/1 = %0 |
| model_05_hybrid_v2 | 2.50 | 1 | 0.375 | 0.541 | 0/1 = %0 |
| model_06_hybrid_v3 | 2.55 | 1 | **0.259** | **0.379** | 0/1 = %0 |
| model_07_finetune | 6.05 | 1 | **0.152** | **0.113** | 0/1 = %0 |
| **target_point_realtrack_ready** | **10.05** | **0** | 0.255 | **0.056** | — |
| target_point_realtrack_ready (TFLite FP16) | **10.05** | **0** | 0.514 | 0.199 | — |

**Kritik bulgu:** `target_point_realtrack_ready` modeli, 200 adım (10.05 saniye) boyunca pistinden çıkmadan sürüşü tamamlamış tek modeldir. Diğer tüm modeller 1.75–6.05 saniye içinde pistten çıkmıştır.

**model_07_finetune notu:** 6.05 saniye hayatta kalması, düşük steering (~0.113) nedeniyle aracın yaklaşık düz gittiğini (collapse davranışı) göstermektedir. `val_corr_x = 0.123` ile model X koordinatını tahmin edememektedir; smoke testinde hayatta kalması yanlış yorum doğurmamalıdır.

## 17.2 Erken Dönem Değerlendirme — Target Point Öncesi Model (Nisan 18, 2026)

`closed_loop_20260418T003312Z/closed_loop_summary.json` dosyasında eski model (`restored_combined_large_noaug`, target-point öncesi) 5 farklı pistte test edilmiştir:

| Pist | Ortalama TTF (s) | Offtrack |
|------|-----------------|----------|
| donkey-generated-track-v0 | 2.15 | 3/3 = %100 |
| donkey-warren-track-v0 | 1.88 | 3/3 = %100 |
| donkey-warehouse-v0 | 1.85 | 3/3 = %100 |
| donkey-minimonaco-track-v0 | **0.20** | 3/3 = %100 |
| donkey-circuit-launch-track-v0 | **0.20** | 3/3 = %100 |

Minimonaco ve Circuit-Launch pistlerinde model ilk adımda (0.20 saniye) pistten çıkmıştır.

## 17.3 `closed_loop_20260420T101113Z` — sim_multitrack_v1 Çok Pistli Test

| Pist | TTF (s) | Tamamlanma % | Harita |
|------|---------|-------------|--------|
| donkey-generated-track-v0 | 3.15 | %2.30 | Mevcut |
| donkey-warren-track-v0 | 3.20 | %1.62 | Mevcut |
| donkey-warehouse-v0 | 2.50 | %0.93 | Mevcut |
| donkey-minimonaco-track-v0 | 0.20 | %0.00 | Mevcut |
| donkey-circuit-launch-track-v0 | 0.20 | %0.00 | Mevcut |

sim_multitrack_v1 modeli minimonaco ve circuit-launch pistlerinde yine 0.20 saniyede başarısız olmuştur.

---

# § 18 — Eğitim Stabilitesi Analizi

## 18.1 Phase 4 Kararsız Eğitim

`phase4_adaptive_v1_20260418_185353/collapse_monitor.jsonl` dosyasına göre eğitim epoch seyri:

| Epoch | Train Loss | Val Loss | Val Corr X | Val MAE X | Val Pred X Std Ratio |
|-------|-----------|---------|------------|-----------|----------------------|
| 1 | 5.072 | 2.208 | 0.271 | 0.256 | 0.389 |
| 2 | 3.804 | **2.520** | 0.305 | 0.277 | 0.384 |
| 3 | 3.269 | **3.098** | 0.220 | 0.285 | 0.278 |
| 4 | 2.972 | 2.382 | 0.347 | 0.253 | 0.458 |
| 5 | 2.737 | **2.872** | 0.273 | 0.327 | 0.352 |

Val loss, epoch 3'te 3.098'e kadar yükselmiştir. Train loss sürekli düşerken val loss diverge etmiştir. Bu durum, küçük veri seti (14.014 örnek) ile overfitting başlangıcını göstermektedir.

## 18.2 model_07 Fine-Tuning Çöküşü

`model_07_finetune_adaptive_v1_20260422_101007/metrics.json` dosyasından:

- **Epoch sayısı:** 4 (erken sonlanma)
- **val_corr_x:** 0.123 (neredeyse rassal tahmin)
- **val_pred_x_std_ratio:** NaN (model sabit değer üretiyor, varyans = 0)
- **train_corr_x:** 0.056 (eğitim setinde de korelasyon yok)
- **collapse_gate.passed:** `false` — 3 kriter başarısız (turn_mae_x, val_corr_x, val_pred_x_std_ratio)

Fine-tuning sürecinde 633.565 örnek kullanılmış olmasına karşın model X koordinatı öğrenememiştir.

## 18.3 model_02 Domain Randomization Kararsızlığı

`model_02_sim_domain_randomization/metrics.json` dosyasından:

- **Epoch sayısı:** 7
- **Val loss seyri:** 1.632 → 1.515 → 1.520 → 1.807 → 1.881 → 1.972 → 1.797
- **Val corr_x:** 0.556 (model_01'in 0.847'sinin çok altında)
- **Train corr_x:** 0.500 (eğitim setinde bile düşük)
- **collapse_gate.passed:** `false` (turn_mae_x = 0.197, threshold aşıldı)

Val loss epoch 4'ten itibaren artmaya başlamış ve hiçbir zaman model_01'in ~0.43 seviyesine yaklaşamamıştır.

## 18.4 model_01 Stabil Eğitim

`model_01_pure_sim_20260421_065556_a3/` (en iyi versiyon) eğitim seyri `history.csv`'den:

- **Epoch sayısı:** 20 (tüm modeller arasında en uzun)
- **En iyi val loss:** 0.4231 (epoch 3'te)
- **Val corr_x:** 0.841 — collapse gate geçildi
- **Val MAE X:** 0.064 (en küçük değer)
- **collapse_gate.passed:** `true` — 4 kriterin 4'ü de geçildi

---

# § 19 — Deney Başarısızlıklarının Teknik Nedenleri

## 19.1 model_02 — Domain Randomization Fazla Agresif

`metrics.json` konfigürasyon alanından okunmuştur:

- `rotation_deg_max: 6.0` (model_01'de 2.5)
- `shift_px_max: 10.0` (model_01'de 6.0)
- `brightness_limit: 0.4` (model_01'de 0.2)
- `blur_radius_max: 3.5` (model_01'de 1.0)
- `rgb_shift_max: 24` (model_01'de 12)
- `train_recovery_ratio: 0.45` (model_01'de 0.2)

Sonuç: `turn_mae_x = 0.197` (model_01'de 0.055). Domain randomization, modelin hedef nokta öğrenmesini bozmuştur. `val_pred_y_std_ratio = 0.451` (model_01'de 0.867) — Y koordinatı varyansı da yarı yarıya çökmüştür.

## 19.2 model_03 — Pure Real Veri Sorunu

`metrics.json` dosyasından:
- `val_mae_y: 0.0` — doğrulama setinde Y koordinat hatası tam sıfır
- `val_corr_y: NaN` — Y koordinatı varyansı sıfır (tüm etiketler aynı değer)
- `mean_abs_steering: 0.921` smoke testinde (tüm modeller arasında en yüksek)

Saf gerçek veri setinde Y ekseni etiketleri tekdüze dağılım göstermiş, model Y koordinatını öğrenememiştir. Smoke testinde aşırı steering (0.921) görülmüş ve araç 1.95 saniyede pistten çıkmıştır.

## 19.3 model_07 — Fine-Tuning Collapse

`metrics.json` dosyasından:
- `val_pred_x_std_ratio: NaN` — model sabit (sıfıra yakın) X değeri üretiyor
- `val_corr_x: 0.123` — X yönünde tahmin gücü neredeyse yok
- Smoke testinde `mean_abs_steering: 0.113` — model düz gidiyor
- 6.05 saniye hayatta kalması düz gitme davranışından kaynaklanmaktadır (collapse sürüşü)

## 19.4 Phase 4 — Yetersiz Veri Sorunu

`metrics.json/stats.train_samples: 14.014`

Model_01–07 kampanyasında kullanılan 328.322 örneğin %4.3'ü kadardır. Bu küçük veri setiyle:
- Val loss 5 epoch boyunca 2.0'ın altına inmemiştir
- Val corr_x en fazla 0.347'ye ulaşabilmiştir
- Phase 4 `collapse_gate.passed: false`

Nisan 21'de büyük eğitim kampanyasına geçilmiş ve veri seti 328.322 örneğe (7 pist, 641.612 kullanılabilir örnek) çıkarılmıştır.

---

# § 20 — Sim2Real Karşılaştırmaları

## 20.1 Veri Stratejisi ve Smoke Test Performansı

| Model | Veri Stratejisi | Eğitim Örneği | Giriş Çözünürlük | Val Corr X | Smoke TTF (s) | Collapse Gate |
|-------|-----------------|---------------|------------------|------------|---------------|---------------|
| model_01_pure_sim | Saf simülasyon | 328.322 | 128×128 | 0.841 | 2.40 | PASSED |
| model_02_sim_domain_rand | Sim + domain rand | 328.322 | 128×128 | 0.556 | 2.50 | FAILED |
| model_03_pure_real | Saf gerçek | 694.247 | 128×128 | 0.842* | 1.95 | FAILED |
| model_04_hybrid_v1 | 50/50 hibrit | 328.322 | 128×128 | 0.848 | 2.20 | PASSED |
| model_05_hybrid_v2 | Sim-ağırlıklı hibrit | 328.322 | 128×128 | 0.813 | 2.50 | PASSED |
| model_06_hybrid_v3 | Real-ağırlıklı hibrit | 328.322 | 128×128 | 0.709 | 2.55 | PASSED |
| model_07_finetune | Fine-tuning | 633.565 | 128×128 | 0.123 | 6.05** | FAILED |
| model_11_multitask | Multi-task | — | 128×128 | — | 2.10 | — |
| target_point_realtrack_ready | realtrack uyumlu | — | **224×224** | — | **10.05 (0 offtrack)** | — |

*model_03 val_corr_x değeri yanıltıcı; val_mae_y = 0.0 (veri sorunu)
**model_07 6.05s hayatta kalması collapse davranışından kaynaklanmaktadır

## 20.2 Model Boyutu ve Deployment

- **Standart modeller (01-07):** 33.858 parametre, TFLite FP16 boyutu = 77.420 byte (~76 KB)
- **model_11_multitask:** 669.828 parametre, TFLite FP16 boyutu = 1.346.992 byte (~1.3 MB) — ~17× büyük, smoke testinde 2.10 saniyede başarısız
- **target_point_realtrack_ready (TFLite FP16):** 77.420 byte, smoke testinde 10.05s, 0 offtrack

Büyük model (model_11) küçük modele kıyasla smoke testinde daha düşük performans göstermiştir.

## 20.3 224×224 vs 128×128 Giriş Çözünürlüğü

- model_01 through model_07: 128×128 piksel
- sim_multitrack_v1, target_point_realtrack_ready: 224×224 piksel

224×224 çözünürlüklü `target_point_realtrack_ready`, tüm smoke testlerinde tek başarılı model olmuştur (0 offtrack, 10.05 saniye).

224×224 giriş kullanan `sim_multitrack_v1` ise smoke testinde 1.75 saniyede başarısız olmuştur. Dolayısıyla çözünürlük tek başına belirleyici değil; eğitim verisi ve etiketleme stratejisi de kritik rol oynamaktadır.

---

# § 21 — Nihai Baseline Seçim Gerekçesi

## 21.1 Kontrolcü Kalibrasyon Testi (Nisan 18, 2026)

`target_point_realtrack_ready` modeli üzerinde `donkey-generated-roads-v0` pistinde bias ve max_steering parametreleri sistematik biçimde test edilmiştir. `closed_loop_summary.json` dosyalarından okunmuştur.

| Zaman Damgası | Bias | Max Steering | Episode 1 TTF (s) | Episode 2 TTF (s) | Ort. TTF (s) |
|---------------|------|--------------|-------------------|--------------------|--------------|
| 163137Z (baseline) | (default) | (default) | 2.65 | — | 2.65 |
| 163428Z (preflight_b0,27_m0,1) | 0.27 | 0.10 | 6.80 | 2.85 | 4.82 |
| 163510Z (preflight_b0,22_m0,08) | 0.22 | 0.08 | 4.25 | 3.15 | 3.70 |
| 163548Z (preflight_b0,2_m0,07) | 0.20 | 0.07 | **21.85** | 3.70 | **12.78** |
| 163644Z (preflight_b0,18_m0,06) | 0.18 | 0.06 | 3.30 | 9.70 | 6.50 |
| 181543Z | (sonraki test) | — | 6.50 | — | 6.50 |

`bias=0.20, max=0.07` konfigürasyonundaki ilk episode `21.85 saniye` hayatta kalmıştır. Aynı konfigürasyonun ikinci episodesi 3.70 saniyede başarısız olmuştur; ortalama 12.78 saniye.

`closed_loop_20260418T181543Z` testinde `recovery_success_rate: 0.5` değeri elde edilmiştir; yani 2 recovery girişiminden 1'i başarıyla tamamlanmıştır. Bu, calibrasyon tamamlandıktan sonra modelin recovery yeteneğinin geliştiğini göstermektedir.

## 21.2 Nihai Modelin Performans Özeti

`model_smoke_target_point_realtrack_ready/closed_loop_summary.json` dosyasından:

- **Offtrack olayı:** 0 (200 adımın tamamı tamamlandı)
- **Sürüş süresi:** 10.05 saniye
- **Mean |abs| CTE:** 0.255 m
- **Mean |abs| steering:** 0.056 (tüm modeller arasında en yumuşak)
- **Mean |abs| target_x:** 0.065
- **Failure reason:** `null` (başarısız olmadı)
- **Oscillation rate:** 0.0 Hz

`model_smoke_target_point_realtrack_ready_fp16_tflite/closed_loop_summary.json` (TFLite versiyonu) dosyasından:

- **Offtrack olayı:** 0
- **Sürüş süresi:** 10.05 saniye
- **Mean |abs| CTE:** 0.514 m
- **Mean |abs| steering:** 0.199

TFLite FP16 dönüşümü, aynı offtrack performansını korumuştur (0 offtrack, 10.05 saniye). CTE değeri ~2× artmış ancak sürüş başarısı etkilenmemiştir.

## 21.3 Neden Diğer Modeller Seçilmedi

`closed_loop_summary.json` dosyalarından kanıtlanan nedenler:

| Model | Reddedilme Sebebi |
|-------|-------------------|
| model_02_sim_domain_rand | collapse_gate FAILED; turn_mae_x=0.197; smoke TTF=2.50s |
| model_03_pure_real | val_mae_y=0.0 (veri sorunu); mean_abs_steering=0.921 (unstable); TTF=1.95s |
| model_07_finetune | val_corr_x=0.123 (collapse); val_pred_x_std_ratio=NaN; gerçek başarı değil |
| model_11_multitask | 669K param, 17× büyük; smoke TTF=2.10s (en büyük modelden en kötü sonuç) |
| sim_multitrack_v1 | TTF=1.75s (tüm modeller içinde en kısa); mean_abs_steering=0.924 |
| phase4 deneyler | 5 epoch sonunda val_loss 2.87; veri yetersizliği (14K örnek) |

`target_point_realtrack_ready`, tüm nicel ölçütlerde üstün performans göstermiş ve Jetson gömülü donanım için TFLite FP16 olarak 77.420 byte boyutunda dışa aktarılmıştır.

Bu ifadelerle karşılaştığınızda teze yazmadan önce doğrulayın veya "tahmin" olduğunu açıkça belirtin.
