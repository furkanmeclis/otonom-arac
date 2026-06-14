# 02 — Geliştirme Aşamaları

Commit geçmişi ve dosya değişimleri incelenerek geliştirme süreci anlamlı aşamalara gruplandırılmıştır. Aşama sınırları, kod değişiminin niteliğine ve büyüklüğüne göre belirlenmiştir.

---

## Aşama 1: Proje Altyapısının Kurulması

**Tarih Aralığı:** 2026-03-05

**İlgili Commitler:**
- `eee7f31` — first commit
- `c5b68e3` — Enhance training script with device configuration

### Yapılan İşler

DonkeyCar (gym-donkeycar) çerçevesi temel alınarak proje oluşturuldu. İlk commit büyük olasılıkla mevcut bir DonkeyCar kurulumunun repoya aktarılmasından oluşmaktadır (`__MACOSX/` artefaktları bunu destekler; bu dosyalar macOS'ta ZIP'ten çıkarılan dosyaların izleridir). 

İlk commit içeriği:
- `manage.py`: Araç yönetim betiği
- `train.py`: Eğitim betiği (temel haliyle)
- `config.py`: 770 satırlık kapsamlı yapılandırma dosyası
- `myconfig.py`: Kullanıcıya özgü yapılandırmalar
- `calibrate.py`: Araç kalibrasyonu
- `data/`: İlk veri kümesi örnekleri
- `models/`: İlk modeller (`dilara.h5`, `my_first_pilot.h5`, `my_2_first_pilot.h5`)

İkinci committe `train.py`'e GPU/CPU seçimi eklendi ve `requirements-train.txt` oluşturuldu.

### Teknik Değerlendirme

Bu aşama, altyapı kurulumuna karşılık gelir. Proje sıfırdan başlatılmamış; mevcut DonkeyCar çerçevesi üzerine inşa edilmiştir. İlk modellerin (`dilara.h5` vb.) varlığı, committen önce yerel denemeler yapıldığına işaret etmektedir.

### Tezde Kullanılabilecek Anlatım

Projenin geliştirilmesinde DonkeyCar açık kaynak çerçevesi temel alınmıştır. Bu çerçeve, araç kontrolü, veri toplama ve model eğitimi için hazır bileşenler sunmaktadır. İlk aşamada çerçeve yerel ortama kurulmuş ve temel eğitim altyapısı doğrulanmıştır.

---

## Aşama 2: Hedef Nokta Modülünün Geliştirilmesi

**Tarih Aralığı:** 2026-03-11 — 2026-03-24

**İlgili Commitler:**
- `b28d47b` — Add lane_v1 model configuration
- `6a8538a` — feat: Add target-point training and inference module

### Yapılan İşler

Projenin en kritik mimari kararı bu aşamada alındı: doğrudan direksiyon tahmini yerine **hedef nokta tahmini** yaklaşımına geçiş. `target_point/` modülü sıfırdan oluşturuldu:

| Dosya | İşlev |
|-------|-------|
| `target_point/model.py` | CNN modeli (hedef nokta tahmini) |
| `target_point/controller.py` | Hedef noktadan direksiyon hesaplama |
| `target_point/dataset.py` | Veri kümesi yükleme ve işleme |
| `target_point/training.py` | Eğitim döngüsü ve metrikler |
| `target_point/diagnostics.py` | Teşhis araçları |
| `target_point/pilot.py` | Gerçek zamanlı çıkarım |

Ayrıca `tests/test_target_point.py` ile birim testleri eklendi.

### Teknik Değerlendirme

Commit `6a8538a`, 1.348 satır ekleme ile projenin en büyük tek özellik committir. Commit mesajı, bu committe tüm ana bileşenlerin birlikte eklendiğini açıkça belirtmektedir. Bu, modülün uzun bir süre boyunca yerel olarak geliştirilip tek seferde commitlendiğine işaret etmektedir.

### Tezde Kullanılabilecek Anlatım

Hedef nokta tabanlı kontrol yaklaşımında model, kamera görüntüsünden doğrudan direksiyon açısı değil, ego-frame koordinat sisteminde ilerleyen bir noktanın `(x, y)` koordinatını tahmin eder. Bu ara temsil, simülasyon ile gerçek dünya arasındaki görsel domain farkına karşı daha dayanıklı bir yapı oluşturmaktadır. Tahmin edilen koordinat, geometrik bir kontrolcü tarafından direksiyon ve gaz komutlarına dönüştürülmektedir.

---

## Aşama 3: Pist Haritalama ve Deneysel Aşamalar (Phase 4–5)

**Tarih Aralığı:** 2026-03-27 — 2026-04-01

**İlgili Commitler:**
- `5850afb` — Add track mapping functionality
- `e02951f` — Add evaluation reports for phase 55 fixed bootstrap
- `ac4920b` — Add new target-point training and evaluation components
- `10cf4b9` — Add configuration for single-track stabilization
- `2952627` — last
- `e9f3908` — Remove unitylog.txt

### Yapılan İşler

**Pist haritalama sistemi** geliştirildi. 6 farklı DonkeyCar pisti için:
- `raw_trace.csv`: Ham sürüş izi verileri
- `centerline.csv`: Merkez hat hesaplaması
- `labels_adaptive_v1.csv`: Adaptif lookahead ile etiketler
- `labels_fixed_1p2m.csv`: 1.2 metre sabit lookahead ile etiketler

Phase 4 ve Phase 5 kapsamında çok sayıda deney yürütüldü:
- **Phase 4:** `adaptive_compare`, `adaptive_flip`, `adaptive_full`, `adaptive_generalize`, `adaptive_recover`, `fixed_compare`, `fixed_flip`, `fixed_full`, `fixed_generalize`, `fixed_recover`, `fixed_run`
- **Phase 5:** `adaptive_hybrid_applied`, `adaptive_robust`, `fixed_robust`

### Teknik Değerlendirme

Commit `5850afb` projenin en büyük commiti olup 43.000+ satır ekleme içermektedir. Bu, uzun bir yerel geliştirme döneminin tek seferde commitlendiğine işaret etmektedir. Deneyler "fixed" (sabit lookahead) ve "adaptive" (adaptif lookahead) olmak üzere iki ana strateji üzerine yapılandırılmıştır.

### Tezde Kullanılabilecek Anlatım

Hedef noktaların hesaplanmasında iki farklı lookahead stratejisi karşılaştırılmıştır: sabit mesafe (1.2 m) ve adaptif mesafe. Adaptif yöntemde hedef nokta mesafesi pist eğriliğine göre dinamik olarak ayarlanmaktadır. Bu iki strateji birden fazla pist üzerinde sistematik biçimde değerlendirilmiştir.

---

## Aşama 4: İleri Denemeler ve Geri Alma

**Tarih Aralığı:** 2026-04-03

**İlgili Commitler:**
- `7caadc8` — revert: roll back to phase5 adaptive robust baseline
- `31f73db` — Remove obsolete diagnostics and label sample files
- `2e9d6e4` — Remove closed loop episode and summary reports

### Yapılan İşler

Commit `7caadc8`, projenin en önemli geri alma işlemini temsil etmektedir. Bu reverte göre; `5850afb`'den sonra geliştirilen aşağıdaki modüller kaldırılmıştır:

| Silinen Modül | Amaç (Tahmini) |
|---------------|----------------|
| `target_point/dagger.py` | Dataset Aggregation (DAgger) algoritması |
| `target_point/scripted_expert.py` | Kurallı uzman politika |
| `target_point/promotion.py` | Model promosyon mekanizması |
| `target_point/temporal.py` | Zamansal model (birden fazla kare) |
| `target_point/effective_loss.py` | Özel kayıp fonksiyonu |
| `run_target_point_dagger.py` | DAgger çalıştırma betiği |
| `run_single_track_stabilization.py` | Tek-pist stabilizasyon betiği |
| `generate_scripted_target_point_data.py` | Scripted expert veri üretimi |
| `analyze_scripted_expert_run.py` | Scripted expert analiz aracı |

Ayrıca Phase 5'in deneysel varyantları (signflip, dynamic, oldstyle, smoke) ve ilgili model dosyaları silindi.

### Teknik Değerlendirme

Bu reverte, birden fazla ileri tekniğin (DAgger, zamansal model, özel kayıp, scripted expert) denendikten sonra elde edilen sonuçların tatmin edici bulunmadığını ve daha sade bir baseline'a geri dönüldüğünü göstermektedir. Commit mesajı açıkça "phase5 adaptive robust baseline"a geri dönüldüğünü belirtmektedir.

### Tezde Kullanılabilecek Anlatım

Geliştirme sürecinde Dataset Aggregation (DAgger), zamansal model ve scripted expert gibi ileri yöntemler denenmiştir. Kod değişiminden anlaşıldığı kadarıyla bu denemelerin sonuçları yeterli görülmemiş ya da ek karmaşıklık getirilmesine değmeyeceği değerlendirilmiş ve daha sade bir mimari üzerinde ilerleme kararı alınmıştır.

---

## Aşama 5: Yeni Pist Denemeleri ve Kapalı Döngü Değerlendirme

**Tarih Aralığı:** 2026-04-05 — 2026-04-08

**İlgili Commitler:**
- `0f0b351` — Add new target point models and mapping utilities for road generation
- `8231732` — Add five run summary report for generated roads validation
- `41389c3` — Son yapılan alan
- `a4c40f7`, `fcb6f8f`, `233ad5a` — Closed loop reports
- `43894f0`, `4f802fc`, `7236feb` — Dokumentasyon
- `2c25c81` — Unity log

### Yapılan İşler

Reverten sonra yeni bir yol gerçekleştirilerek üretilmiş (generated) yollar üzerinde değerlendirme yapıldı. 5 koşu özet raporu oluşturuldu. Kapalı döngü (closed-loop) değerlendirme framework'ü kullanılarak birden fazla sürüş oturumu raporlandı.

Bu aşamada aynı zamanda kapsamlı komut dokümantasyonu (`COMMANDS.md`) ve kurulum kılavuzu (`SETUP.md`) yazıldı.

### Teknik Değerlendirme

Kapalı döngü değerlendirme (`evaluate_closed_loop.py`), modelin simülatörde gerçek koşullar altında performansını ölçmektedir. Birden fazla oturum raporunun art arda commitlenmesi, aktif bir değerlendirme dönemini yansıtmaktadır.

---

## Aşama 6: Mimari Yeniden Yapılanma ve Çok Pist Sim2Real

**Tarih Aralığı:** 2026-04-16 — 2026-04-17

**İlgili Commitler:**
- `c8e688f` — Multi-track Sim2Real guidelines
- `c98a637` — Script to run target-point models for 3 laps
- `5de581c` — Add new data artifacts, models, documentation
- `40d035d` — Refactor: Remove supervised learning scripts
- `95dafd8` — Remove obsolete JSON reports and model files

### Yapılan İşler

**Büyük mimari değişiklik:** `target_point/` modülü `ai_pipeline/target_point/` altına taşındı. Supervised learning örnek kodları (`examples/supervised_learning/`) tamamen kaldırıldı.

Yeni bileşenler eklendi:
- `ai_pipeline/target_point/export.py`: TFLite dışa aktarma
- `ai_pipeline/target_point/external_adapter.py`: Harici veri kümesi dönüştürücü
- `ai_pipeline/target_point/pilot_tflite.py`: Jetson için hafif çıkarım

### Teknik Değerlendirme

`examples/supervised_learning/` kaldırılması, projenin artık genel DonkeyCar supervised learning yaklaşımından tamamen uzaklaştığını ve target-point yöntemine odaklandığını göstermektedir. TFLite desteği, gerçek donanım (Jetson) üzerinde dağıtım hedefine işaret etmektedir.

---

## Aşama 7: Büyük Ölçekli Veri Toplama ve Çok Model Eğitimi

**Tarih Aralığı:** 2026-04-18 — 2026-04-22

**İlgili Commitler:**
- `6687c15` — Unity log
- `2995dab` — Bias compensation and dataset collection scripts
- `ec10b31` — Update simulation configs and add model training scripts
- `3f66a3f` — Multi-task and fine-tuned models

### Yapılan İşler

Kontrolcüye **bias kompansasyonu** ve **deadband** parametreleri eklendi. Büyük ölçekli simülasyon veri toplama betikleri oluşturuldu (`collect_massive_sim_dataset.ps1`).

Model eğitim planı sistematik hale getirildi; 12 model konfigürasyonu tanımlandı:

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

Batch size 32'den 128'e yükseltildi.

### Teknik Değerlendirme

Bu aşama, projenin en sistematik deneysel aşamasıdır. Farklı veri karışımı stratejileri (pure sim, pure real, hybrid) karşılaştırmalı olarak değerlendirilmektedir. Multi-task ve temporal model denemeleri, mimarinin genişletilmesine yönelik araştırmacı ilgisini göstermektedir.

### Tezde Kullanılabilecek Anlatım

Sim-to-real transfer problemini araştırmak amacıyla birden fazla model konfigürasyonu sistematik olarak karşılaştırılmıştır. Bu konfigürasyonlar, yalnızca simülasyon verisi kullanan modellerden yalnızca gerçek veri kullanan modellere uzanan bir spektrumu kapsamakta; ayrıca farklı oranlarda karma (hybrid) veri stratejilerini de içermektedir.

---

## Aşama 8: Dokümantasyon ve Yayına Hazırlık

**Tarih Aralığı:** 2026-04-26

**İlgili Commitler:**
- `0ce7fd9` — Add detailed project documentation
- `4ba0252` — fix: repo klonlanabilirliğini düzelt
- `d483ea6` — fix: update paths in documentation
- `b27709b` — refactor: update README
- `358d90e` — fix: hardcoded kişisel yolları kaldır
- `7466d2b` — fix: 4 tutarsızlığı düzelt
- `fec3302` — fix: kalan küçük tutarsızlıkları gider

### Yapılan İşler

Tek günde 7 commit: kapsamlı proje dokümantasyonu yazıldı, repo başkalarının klonlayabileceği hale getirildi (hardcoded yollar kaldırıldı, tutarsızlıklar giderildi, README güncellendi).

### Teknik Değerlendirme

Bu commitler yoğunluğu, projenin bu noktada "teslime hazır" hale getirildiğine güçlü biçimde işaret etmektedir. Hardcoded yolların kaldırılması ve klonlanabilirlik düzeltmeleri, projenin başka ortamlarda veya kişiler tarafından çalıştırılacağı beklentisini yansıtmaktadır.
