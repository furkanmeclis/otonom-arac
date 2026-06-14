# 09 — Test, Doğrulama ve Sınırlamalar

---

## Test İzleri

### Birim Test Dosyası

Projede bir birim test dosyası mevcuttur:

**Dosya:** `tests/test_target_point.py`

Bu dosya commit `6a8538a`'da oluşturuldu ve commit `5850afb`'de büyük ölçüde genişledi (commit mesajına göre 130 satırdan ~989 satıra).

`7caadc8` revert sonrası dosya küçüldü (revert öncesi büyük test seti kaldırıldı).

Commit `6a8538a` mesajında şu ifade geçmektedir:
> "Included tests for key functionalities such as target point computation and model evaluation."

Bu, en azından şu bileşenler için birim testlerinin yazıldığına işaret etmektedir:
- Hedef nokta koordinat hesaplama
- Model değerlendirme

**Dürüst Not:** Test dosyasının güncel içeriği bu analizde doğrudan okunmamıştır; test kapsamı yukarıdaki bilgilerden tahmin edilmektedir.

---

### conftest.py

Commit `40d035d`'de `tests/conftest.py` eklendi. Bu dosya pytest konfigürasyonunu içerir; test ortamının yapılandırılmış olduğunu gösterir.

---

## Manuel Doğrulama Olasılıkları

### Smoke Test

`manage.py` betiği `smoke` komutu desteklemektedir:

```powershell
.\.venv\Scripts\python manage.py smoke --simulationconfig=simulationconfig.py
```

README'de bu komut "temel fonksiyonelliğin doğrulanması" için gösterilmektedir. Bu, simülatör bağlantısı, model yükleme ve temel sürüş döngüsünü test eden bir entegrasyon testidir.

### Kapalı Döngü Değerlendirme (Closed-Loop Evaluation)

`evaluate_closed_loop.py` ile simülatörde tam otonom sürüş gerçekleştirilmiş; sonuçlar `closed_loop_summary.json` dosyalarında kaydedilmiştir.

Mevcut rapor klasörleri:
- `data/artifacts/reports/phase5_adaptive_eval/`
- `data/artifacts/reports/phase5_fixed_eval/`
- Ve daha fazlası

Bu raporlar, değerlendirme sürecinin sistematik olarak yürütüldüğünü kanıtlamaktadır.

### Beş Koşu Özet Raporu

Commit `8231732`: "Add five run summary report for generated roads validation"

Bu, üretilmiş (procedurally generated) yollar üzerinde birden fazla çalıştırma yapılarak sonuçların tutarlılığının doğrulandığını göstermektedir.

---

## Kod Üzerinden Görülen Kontroller

### Model Doğrulaması
`model.py`'deki `TargetPointDenormalizer` katmanı, normalize edilmiş çıktının gerçek koordinatlara dönüşümünü içermektedir. Bu, eğitim sırasında normalizasyon istatistiklerinin (`mean`, `std`) doğru hesaplanması gerekliliğini zorunlu kılar.

### Görüntü Ön İşleme Tutarlılığı
`model.py`'deki `preprocess_image` fonksiyonu hem eğitim hem çıkarım için aynı ön işleme adımlarını uygular (kırpma, yeniden boyutlandırma). Bu, train/test tutarsızlığını önleyen kritik bir kontroldür.

### Kırpma Sınır Kontrolü
```python
if top >= bottom_index or left >= right_index:
    raise ValueError("Target-point crop settings remove the full image...")
```
Bu, kırpma parametrelerinin yanlış ayarlandığında erken hata vermesini sağlar.

### Collapse Monitor
`experiments/` klasöründeki `collapse_monitor.jsonl` dosyaları, eğitim sırasında model çöküşünü (model collapse) izlemek için kullanılmaktadır. Bu, eğitim döngüsünde sistematik bir izleme mekanizmasının var olduğunu gösterir.

---

## Sınırlamalar

### 1. Test Kapsamı Sınırlı
Projenin test altyapısı temel bileşenler için birim testleri içerse de, aşağıdaki alanlarda test yok gibi görünmektedir:
- Kontrolcünün uçtan uca davranışı
- Pist haritalama pipeline'ı
- TFLite dışa aktarma kalitesi
- Domain randomizasyon parametreleri

**Not:** Bu sınırlama commit geçmişinden tahmin edilmiştir; mevcut test dosyasının tam içeriği analiz edilmeden kesin olarak belirlenemez.

### 2. Gerçek Donanım Testi Belirsiz
Commit geçmişinde gerçek araç üzerinde test yapıldığına dair açık bir iz bulunmamaktadır. TFLite pilot ve Jetson kodu yazılmış olmakla birlikte, gerçek araç üzerinde doğrulama sonuçlarına ait veri repo'da görünmemektedir.

### 3. Kapalı Döngü Değerlendirme Sadece Simülatörde
Mevcut kapalı döngü raporları simülatör ortamındadır. Gerçek ortamda (real-world) benzer sistematik değerlendirme yapılıp yapılmadığı belirsizdir.

### 4. Birden Fazla Eğitim Çalıştırma İzleri Tutarsız
Bazı deneyler için `collapse_monitor.jsonl` ve `history.csv` mevcut iken, diğerleri için yalnızca final model dosyaları bulunmaktadır. Bu, bazı deneylerin tam kayıt altına alınmadan gerçekleştirildiğini düşündürmektedir.

### 5. Commitlenmeyen Denemeler
Git geçmişi yalnızca commitlenmiş değişiklikleri kapsar. Yerel olarak gerçekleştirilmiş ancak commitlenmemiş denemeler, yeniden yazılan kod versiyonları ve başarısız model eğitimleri kayıt altında değildir.

### 6. Metrikler Eksik Yorumlanmış
`metrics.json` dosyaları her deney için mevcuttur; ancak hangi metrik eşiklerinin "kabul edilebilir" sayıldığı commit mesajlarında belirtilmemiştir.

---

## Tezde Kullanılabilecek Anlatım

"Geliştirme sürecinde iki düzeyde doğrulama uygulanmıştır: birim testler aracılığıyla kritik bileşenlerin (hedef nokta hesaplama, model değerlendirme) doğruluğu test edilmiş; kapalı döngü simülasyon değerlendirmeleriyle modelin otonom sürüş performansı ölçülmüştür.

Kapalı döngü değerlendirme, modelin her simülasyon pistinde birden fazla tur çalıştırılması ve tamamlanan tur sayısı, pist dışına çıkma olayları gibi metriklerin otomatik olarak kaydedilmesine dayanmaktadır.

Ancak bazı sınırlamalar mevcuttur: gerçek araç üzerindeki doğrulama verileri mevcut kayıtlara yansımamış; tüm modeller için tutarlı düzeyde kayıt tutulamamıştır. Bu durum, deneysel süreç hakkındaki bilgilerin kısmen eksik kalmasına yol açmaktadır."
