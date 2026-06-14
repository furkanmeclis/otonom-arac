# 03 — Deneyler ve Denemeler

Bu dosyada, Git geçmişinden çıkarılabilen deneme izleri belgelenmiştir. Silinen kodlar, geri alınan commitler ve değiştirilen yapılar analiz edilmiştir. Kesin olmayan yorumlar açıkça işaretlenmiştir.

---

## Deneme 1: DAgger (Dataset Aggregation) Algoritması

### Kanıt
- `target_point/dagger.py` dosyası oluşturulmuştu; `7caadc8` reverte ile silindi.
- `run_target_point_dagger.py` çalıştırma betiği de aynı reverte kaldırıldı.
- Commit `6a8538a`'dan `7caadc8`'e kadar olan dönemde bu dosyaların var olduğu anlaşılmaktadır.

### Ne Denenmiş Olabilir?
DAgger (Dataset Aggregation), imitation learning'de kullanılan iteratif bir yöntemdir. Temel fikir: modeli eğittikten sonra modelin sürüş sırasında düştüğü durumlarda uzman müdahalesi yaparak yeni veri toplamak ve modeli yeniden eğitmektir. Revert öncesinde `target_point/dagger.py` dosyasının varlığı, bu yöntemin uygulandığına işaret etmektedir.

### Sonuç
Reverte (`7caadc8`) ile kaldırıldı.

### Neden Değiştirilmiş Olabilir?
Commit mesajı "phase5 adaptive robust baseline"a dönüş olduğunu belirtmektedir. Muhtemelen DAgger'ın getirdiği ek karmaşıklık (iteratif döngü, uzman müdahalesi gerekliliği), elde edilen performans kazanımıyla orantılı görülmedi. Kesin neden commit mesajında belirtilmemiştir.

### Tezde Nasıl Anlatılabilir?
"Geliştirme sürecinde Dataset Aggregation (DAgger) yöntemi denenmiş; ancak kod değişiminden anlaşıldığı kadarıyla bu yöntem belirli bir aşamada kaldırılmış ve daha sade bir supervised learning pipeline'ına geri dönülmüştür."

---

## Deneme 2: Scripted Expert (Kurallı Uzman Politika)

### Kanıt
- `target_point/scripted_expert.py` — `7caadc8` ile silindi.
- `generate_scripted_target_point_data.py` — aynı reverte silindi.
- `analyze_scripted_expert_run.py` — aynı reverte silindi.

### Ne Denenmiş Olabilir?
Scripted expert, kurallı (kural tabanlı) bir araç kontrolcüsü kullanarak otomatik veri üretme yaklaşımıdır. Gerçek sürücüden bağımsız olarak büyük miktarda etiketli veri üretmek amacıyla kullanılmış olabilir. `generate_scripted_target_point_data.py` betiğinin varlığı bunu desteklemektedir.

### Sonuç
Tüm bileşenler reverte ile kaldırıldı.

### Neden Değiştirilmiş Olabilir?
Scripted expert ile üretilen verilerin gerçek sürüş verisiyle kalitede rekabet edemediği ya da scripted verilerin domain bias yarattığı değerlendirilmiş olabilir. Kesin gerekçe belirsizdir.

### Tezde Nasıl Anlatılabilir?
"Eğitim verisi artırmak amacıyla kural tabanlı bir uzman politika oluşturularak simülatör üzerinde otomatik veri üretimi denenmiştir. Bu yaklaşım, koda yansıyan izlerden anlaşıldığı üzere, sonradan terk edilerek yalnızca insan sürüşünden toplanan verilerle devam edilmiştir."

---

## Deneme 3: Zamansal Model (Temporal Model)

### Kanıt
- `target_point/temporal.py` — `7caadc8` ile silindi.
- `configs/model_12_temporal.py` — hâlâ mevcuttur (silinmedi).

### Ne Denenmiş Olabilir?
Zamansal model, tek bir kare yerine ardışık kareleri girdi olarak kullanarak hareket bilgisini kodlamayı amaçlamaktadır. `model_12_temporal.py` yapılandırmasının hâlâ mevcut olması ilginçtir; `temporal.py` modülü kaldırılmış olsa bile bu konfigürasyonun varlığı, zamansal yaklaşımın daha ileri dönemde yeniden ele alınmaya planlandığını ya da yalnızca yapılandırma dosyasının gözden kaçtığını düşündürmektedir.

### Sonuç
`temporal.py` modülü kaldırıldı. Konfigürasyon dosyası kaldı.

### Neden Değiştirilmiş Olabilir?
Zamansal modeller, eğitim karmaşıklığını ve bellek gereksinimini artırır. Muhtemelen elde edilen kazanım bu maliyeti karşılamadı.

### Tezde Nasıl Anlatılabilir?
"Zamansal bilgiyi modele entegre etmek amacıyla ardışık kare dizilerini girdi olarak kullanan bir mimari denenmiştir. Bu yaklaşım kod değişiminde izler bırakmış olmakla birlikte, geliştirme sürecinde belirli bir noktada kaldırılmıştır."

---

## Deneme 4: Promotion Mekanizması

### Kanıt
- `target_point/promotion.py` — `7caadc8` ile silindi.

### Ne Denenmiş Olabilir?
Muhtemelen iyi performanslı model checkpoint'lerini "promote" ederek bir sonraki eğitim aşamasına başlangıç noktası olarak kullanan bir mekanizma. İteratif eğitim pipeline'larında kullanılan bu yaklaşım, DAgger veya curriculum learning ile birlikte kullanılıyor olabilir.

### Sonuç
Reverte ile kaldırıldı.

---

## Deneme 5: Özel Kayıp Fonksiyonu (Effective Loss)

### Kanıt
- `target_point/effective_loss.py` — `7caadc8` ile silindi.

### Ne Denenmiş Olabilir?
Standart MSE yerine özel bir kayıp fonksiyonu denenmiş olabilir. Örneğin, pistin eğriliğine göre ağırlıklı kayıp, yanlış tahminleri farklı biçimde penalize eden bir kayıp ya da hedef noktanın yönünü de hesaba katan geometrik kayıp.

### Sonuç
Reverte ile kaldırıldı.

### Tezde Nasıl Anlatılabilir?
"Model eğitiminde standart ortalama kare hata (MSE) kaybına alternatif özel kayıp fonksiyonları araştırılmıştır. Kod değişiminden anlaşıldığı kadarıyla bu alternatifler belirli bir aşamada kaldırılmış ve standart kayıp fonksiyonuna geri dönülmüştür."

---

## Deneme 6: Tek-Pist Stabilizasyon

### Kanıt
- `run_single_track_stabilization.py` — `7caadc8` ile silindi.
- Commit `10cf4b9`: "Add new configuration and script for single-track stabilization"

### Ne Denenmiş Olabilir?
Önce tek bir pist üzerinde modeli stabilize etmeye (güvenilir performans sağlamaya) odaklanıp ardından diğer pistlere genelleme yapma stratejisi denenmiş olabilir. Curriculum learning yaklaşımının basit bir uygulaması olarak yorumlanabilir.

### Sonuç
Reverte ile kaldırıldı.

---

## Deneme 7: Phase 5 Varyantları (Signflip, Dynamic, Oldstyle, Smoke)

### Kanıt
- Commit `2e9d6e4`: "Remove closed loop episode and summary reports for phase 5 signflip dynamic, oldstyle, and smoke"
- Bu raporlar `5850afb` commitinde eklenmiş, `2e9d6e4`'te kaldırılmıştır.

### Ne Denenmiş Olabilir?
- **signflip:** Direksiyon işareti çevirme ile data augmentation deneyi
- **dynamic:** Dinamik lookahead veya dinamik parametre deneyi
- **oldstyle:** Eski stil kontrol yaklaşımı karşılaştırması
- **smoke:** Hızlı sanity check koşusu

### Sonuç
Tüm bu deney sonuçları `2e9d6e4` ile silindi. Phase 5'in "adaptive_robust" ve "fixed_robust" varyantları korundu.

### Tezde Nasıl Anlatılabilir?
"Phase 5 kapsamında birden fazla deney varyantı yürütülmüştür. Bu varyantlar arasından en iyi sonuç veren yapılandırmalar seçilerek devam edilmiş; diğer deney sonuçları repodan temizlenmiştir."

---

## Deneme 8: Adaptif vs. Sabit Lookahead Karşılaştırması

### Kanıt
- `labels_adaptive_v1.csv` ve `labels_fixed_1p2m.csv` dosyaları paralel olarak oluşturuldu.
- Phase 4'te `adaptive_*` ve `fixed_*` olarak adlandırılan deneyler sistematik biçimde karşılaştırıldı.
- Phase 5'te de aynı desen sürdürüldü.

### Ne Denenmiş Olabilir?
Hedef nokta mesafesinin sabit mi (1.2 m) yoksa pist geometrisine göre adaptif mi belirlenmesi gerektiği sorusu araştırılmıştır. Adaptif yaklaşımda düz pistlerde daha uzak, virajlarda daha yakın bir nokta hedeflenmesi amaçlanmış olabilir.

### Tezde Nasıl Anlatılabilir?
"Hedef nokta seçiminde iki farklı strateji karşılaştırmalı olarak incelenmiştir: sabit mesafe yaklaşımı (1.2 m sabit lookahead) ve adaptif mesafe yaklaşımı (pist eğriliğine göre dinamik lookahead). Sistematik deneyler her iki strateji için de eğitim ve değerlendirme verilerinin paralel olarak üretilmesini kapsamıştır."

---

## Deneme 9: Myconfig Varyant Dosyaları (Phase 16, 17, 19)

### Kanıt
- `myconfig_phase16_balanced_v3.py` — `7caadc8` ile silindi.
- `myconfig_phase17_curvature_narrow.py` — `7caadc8` ile silindi.
- `myconfig_phase19_mid_corner_carry.py` — `7caadc8` ile silindi.

### Ne Denenmiş Olabilir?
Farklı myconfig varyantlarının varlığı, çok sayıda eğitim deneyi için farklı yapılandırmaların hızla oluşturulduğuna ve test edildiğine işaret etmektedir. "Balanced", "curvature_narrow", "mid_corner_carry" isimlendirmeleri, kontrolcü veya veri toplama parametrelerindeki farklılıkları yansıtmaktadır.

### Tezde Nasıl Anlatılabilir?
"Geliştirme sürecinde farklı eğitim ve kontrolcü parametreleri için birden fazla yapılandırma dosyası oluşturularak deneyler yürütülmüştür. Bu yöntem, hızlı deney tekrarı (rapid experimentation) yapılmasına olanak tanımıştır."
