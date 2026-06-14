# 05 — Teknik Kararlar

Bu dosyada, Git geçmişi ve mevcut kod yapısından çıkarılabilen teknik kararlar belgelenmiştir.

---

## Teknik Karar 1: Hedef Nokta (Target-Point) Mimarisi

### Karar
Doğrudan direksiyon açısı tahmiri yerine, ego-frame koordinat sisteminde `(target_x, target_y)` tahmin eden bir mimari seçildi.

### Kanıt
- Commit `6a8538a` commit mesajı açıkça bu seçimi belgeler.
- README.md: "Bu yaklaşım sim-to-real transferini kolaylaştırır."
- `target_point/controller.py`: Hedef noktadan geometrik hesaplama ile direksiyon üretilmesi.

### Olası Gerekçe
Görüntüden doğrudan direksiyon tahmiri, simülasyon ile gerçek dünya görsel farklılıklarına (domain gap) duyarlıdır. Hedef nokta tahmiri, bu problemi geometrik bir ara katmana taşıyarak görsel domain farkına karşı daha sağlam bir yapı oluşturur.

### Avantajları
- Simülasyon-gerçek transfer daha kolay
- Geometrik kontrolcü yorumlanabilirlik sağlar
- Farklı pistlere genelleme daha iyi

### Dezavantajları / Sınırlamaları
- Her pist için haritalama (merkez hat hesaplama) gereklidir
- Kontrolcü parametreleri (bias, deadband, gaz profili) ayarlama gerektirir
- Tahmin hatası hem modelden hem kontrolcüden kaynaklanabilir; hata analizi karmaşıklaşır

### Tezde Kullanılabilecek Anlatım
"Klasik uçtan-uca (end-to-end) yaklaşımların aksine, bu çalışmada model çıktısı olarak direksiyon açısı yerine pist üzerinde ilerleyen bir noktanın koordinatı (`target_x`, `target_y`) seçilmiştir. Bu ara temsil katmanı, modelin görsel olmayan bir hedef üzerine öğrenmesini sağlayarak simülasyon-gerçek arası transferi kolaylaştırmaktadır."

---

## Teknik Karar 2: DonkeyCar Çerçevesinin Temel Alınması

### Karar
Projenin altyapısı sıfırdan yazılmadı; DonkeyCar / gym-donkeycar açık kaynak çerçevesi temel alındı.

### Kanıt
- `config.py` (770 satır) ve `manage.py` DonkeyCar kaynaklıdır.
- `gym_donkeycar/` dizini tüm çerçeve kodlarını içermektedir.
- `eee7f31` first commit içeriği büyük oranda DonkeyCar çerçevesidir.

### Olası Gerekçe
Araç kontrolü, simülatör bağlantısı ve temel veri toplama altyapısı hazır bir çerçeveden alınarak geliştirme süresinden tasarruf edildi. Özgün katkı, target-point modülü ve ilgili araçlar üzerinde yoğunlaştırıldı.

### Avantajları
- Araç kontrolü, veri kaydı gibi alt seviyeli problemler önceden çözülmüş
- Topluluk desteği ve belgeleme mevcut

### Dezavantajları / Sınırlamaları
- Çerçeve sınırlamaları (örn. belirli veri formatları) projeyi etkiler
- `config.py` 770 satır ve büyük kısmı kullanılmayan parametreler içerir

---

## Teknik Karar 3: TensorFlow / Keras ile CNN Modeli

### Karar
Model, TensorFlow 2.15.1 / Keras ile geliştirildi. Mimari: depthwise separable convolution tabanlı, ~115K parametreli verimli CNN.

### Kanıt
- `requirements-train.txt` içinde `tensorflow==2.15.1`
- `ai_pipeline/target_point/model.py`: `TargetPointDenormalizer` katmanı ve CNN mimarisi
- README.md: "Efficient CNN (~115K parametre, depthwise separable conv)"

### Olası Gerekçe
- TensorFlow 2.15.1 ve Python 3.11 kombinezonunun seçilmesi, Jetson uyumluluğu veya TFLite desteği için yapıldı.
- Depthwise separable conv, parametre sayısını azaltarak Jetson gibi sınırlı donanımlarda çalışmayı kolaylaştırır.

### Avantajları
- TFLite ile doğrudan dışa aktarma desteği
- Az parametreli model → hızlı çıkarım
- Keras ile hızlı prototipleme

### Dezavantajları / Sınırlamaları
- TF 2.15.1 yalnızca Python 3.11 ile çalışır (README'de belirtilmiş)
- PyTorch ekosistemi ile karşılaştırıldığında bazı araçlar sınırlı

---

## Teknik Karar 4: JSONL Manifest Formatı

### Karar
Veri kümesi için JSONL (JSON Lines) manifest formatı seçildi.

### Kanıt
- `ai_pipeline/target_point/manifest.py` dosyası
- README.md: "Veri Formatı: JSONL manifest + DonkeyCar Tub V2 görüntüleri"

### Olası Gerekçe
JSONL formatı, büyük veri kümeleri için satır satır okuma kolaylığı sağlar; tüm veriyi belleğe yüklemek gerekmez. Ayrıca veri kümelerini karıştırma ve filtreleme işlemleri kolaylaşır.

---

## Teknik Karar 5: Lookahead Stratejisi — Adaptif vs. Sabit

### Karar
İki farklı lookahead stratejisi paralel olarak geliştirildi ve karşılaştırıldı.

### Kanıt
- `labels_adaptive_v1.csv` ve `labels_fixed_1p2m.csv` her pist için paralel üretildi.
- Phase 4 ve Phase 5 deneyleri her iki strateji için tekrarlandı.
- `lookahead_stats_adaptive_v1.json` ve `lookahead_stats_fixed_1p2m.json` istatistikler kaydedildi.

### Olası Gerekçe
Sabit mesafe (1.2 m) uygulaması basittir ancak viraj geometrisine uyum sağlamaz. Adaptif yaklaşım teorik olarak üstündür ancak uygulaması ve parametrizasyonu daha karmaşıktır. Karşılaştırmalı değerlendirme ile hangisinin pratikte daha iyi performans gösterdiği belirlenmek istendi.

### Tezde Kullanılabilecek Anlatım
"Hedef noktası seçiminde iki strateji sistematik olarak karşılaştırılmıştır: 1.2 metre sabit ileri bakış mesafesi (fixed lookahead) ve pist eğriliğine duyarlı adaptif ileri bakış mesafesi (adaptive lookahead). Her iki strateji için eğitim verisi bağımsız olarak üretilmiş ve aynı model mimarisi üzerinde değerlendirilmiştir."

---

## Teknik Karar 6: PowerShell Otomasyon Betikleri

### Karar
Eğitim ve veri toplama süreçleri PowerShell betikleriyle otomatikleştirildi.

### Kanıt
- `scripts/` dizininde birden fazla `.ps1` dosyası
- `train_first6_models.ps1`, `train_models_5_6.ps1`, `train_models_11_12_07.ps1`
- `collect_massive_sim_dataset.ps1`, `monitor_massive_sim_dataset.ps1`
- Commit `7236feb`: "Update command documentation to use PowerShell syntax"

### Olası Gerekçe
Geliştirme ortamı Windows olduğundan PowerShell tercih edilmiştir. Birden fazla modelin sıralı eğitimi için otomasyon zorunlu hale gelmiştir.

---

## Teknik Karar 7: TFLite Dışa Aktarma (Jetson Desteği)

### Karar
Model, TFLite formatında dışa aktarılabiliyor ve Jetson için özel bir hafif pilot (`pilot_tflite.py`) yazıldı.

### Kanıt
- Commit `40d035d`: `export.py` ve `pilot_tflite.py` eklendi
- `ai_pipeline/target_point/export.py`
- `ai_pipeline/target_point/pilot_tflite.py`

### Olası Gerekçe
Fiziksel araç üzerinde çalıştırma hedefi, düşük gecikme ve sınırlı hesaplama kaynakları gerektirmektedir. TFLite, tam TensorFlow'a göre belirgin biçimde daha hızlı çıkarım sağlar.

### Tezde Kullanılabilecek Anlatım
"Gerçek donanım üzerinde dağıtım (deployment) için model TensorFlow Lite formatına dönüştürülmüş ve Jetson platformu için optimize edilmiş hafif bir çıkarım modülü geliştirilmiştir."

---

## Teknik Karar 8: Çok Model Konfigürasyon Sistemi

### Karar
Tek bir model yerine, farklı veri stratejilerini temsil eden 12 model konfigürasyonu sistematik biçimde tanımlandı.

### Kanıt
- `configs/` dizininde `model_01` — `model_12` dosyaları
- `MODEL_TRAINING_PLAN.md` (commit `ec10b31`)
- `model_export_manifest.json` (commit `3f66a3f`)

### Olası Gerekçe
Sim-to-real transfer için hangi veri stratejisinin en iyi performansı verdiği sorusu, ancak karşılaştırmalı deneylerle yanıtlanabilir. Konfigürasyon sistematiği, adil karşılaştırma yapılmasını ve deneylerin tekrar edilebilir olmasını sağlar.

### Tezde Kullanılabilecek Anlatım
"Sim-to-real transfer performansını araştırmak amacıyla, yalnızca simülasyon verisinden yalnızca gerçek veriye uzanan spektrumda 9 farklı model konfigürasyonu tanımlanmış ve sistematik biçimde eğitilmiştir. Bu karşılaştırmalı çerçeve, hangi veri karışımının en iyi genelleme performansını sergilediğini nesnel olarak değerlendirmeye olanak tanımıştır."
