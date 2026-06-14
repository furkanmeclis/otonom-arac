# 04 — Karşılaşılan Problemler ve Çözümler

Bu dosyada commit geçmişi ve kod değişimlerinden çıkarılabilen teknik problemler ve uygulanan çözümler belgelenmiştir.

---

## Problem 1: Simülasyon ile Gerçek Dünya Arasındaki Domain Farkı (Domain Gap)

### Problem Nasıl Anlaşılıyor?
Projenin temel motivasyonu bu probleme dayanmaktadır. README.md'de açıkça belirtilmektedir: "Bu yaklaşım sim-to-real transferini kolaylaştırır." Standart DonkeyCar'ın direksiyon tahmin modellerinin sim-to-real performansının yetersiz kaldığı, hedef nokta yaklaşımına geçişin gerekçesinden anlaşılmaktadır.

### İlgili Commitler
- `6a8538a` — Target-point modülünün oluşturulması (çözüm)
- `c8e688f` — Multi-track Sim2Real guidelines (sistematik yaklaşım)
- `ec10b31` — Model konfigürasyonları: pure sim, pure real, hybrid

### Uygulanan Çözüm
- Doğrudan direksiyon tahmiri yerine hedef nokta (target point) tahririne geçiş.
- Çoklu veri stratejisi (model_01 ile model_06 arası) ile farklı sim/real veri oranları denendi.
- Domain randomizasyon (`domain_randomization.py`) ile simülasyon verisi çeşitlendirildi.
- Bias kompansasyonu ve deadband parametreleri eklendi (commit `2995dab`).

### Çözümün Etkisi
Kod değişiminden anlaşıldığı kadarıyla, hedef nokta yaklaşımı projenin sonuna kadar ana mimari olarak korundu. Bu, yaklaşımın başlangıç hedefini karşıladığına işaret etmektedir.

### Tezde Kullanılabilecek Açıklama
"Simülasyon ortamında eğitilen modellerin gerçek ortamda kullanılmasında görsel domain farkı (domain gap) kritik bir sorun olarak ortaya çıkmaktadır. Bu çalışmada söz konusu sorun, ara temsil olarak hedef nokta kullanımı ve sistematik veri karışımı deneyleriyle ele alınmıştır."

---

## Problem 2: Kontrolcü Bias ve Hassasiyet Sorunu

### Problem Nasıl Anlaşılıyor?
Commit `2995dab`: "Add target-point bias compensation and deadband parameters to controller"

Bias kompansasyonu ve deadband parametresinin eklenmesi, kontrolcünün saf geometrik hesaplama ile gerçek araç davranışı arasında sistematik bir sapma olduğuna işaret etmektedir. Deadband, küçük direksiyon komutlarının filtrelenerek titreşim veya gürültünün önlenmesini sağlar.

### İlgili Commitler
- `2995dab` — Bias kompansasyonu ve deadband eklendi

### Uygulanan Çözüm
`ai_pipeline/target_point/controller.py` dosyasına bias kompansasyonu ve deadband parametreleri eklendi. Bu parametreler `simulationconfig.py` üzerinden ayarlanabilir hale getirildi.

### Çözümün Etkisi
Kontrolcünün daha kararlı direksiyon komutları üretmesi sağlandı.

### Tezde Kullanılabilecek Açıklama
"Geometrik kontrolcünün pratikte uygulanmasında araç davranışından kaynaklanan sistematik sapma (bias) ve düşük genlikli komut titreşimi sorunlarıyla karşılaşılmıştır. Bu sorunlar, kontrolcüye bias kompansasyonu ve deadband filtresi eklenerek giderilmiştir."

---

## Problem 3: Eğitim Verimliliği — Küçük Batch Boyutu

### Problem Nasıl Anlaşılıyor?
Commit `ec10b31`: `TARGET_POINT_BATCH_SIZE` 32'den 128'e yükseltilmiştir.

### İlgili Commitler
- `ec10b31` — Batch size artırıldı

### Uygulanan Çözüm
Batch boyutu 4 kat artırılarak eğitim hızı artırıldı. Bu değişiklik, büyük ölçekli (massive) veri toplama betiklerinin eklenmesiyle eş zamanlı yapıldığından, artan veri miktarının daha büyük batch boyutunu desteklediği anlaşılmaktadır.

### Tezde Kullanılabilecek Açıklama
"Artan veri miktarıyla birlikte eğitim süresini optimum düzeyde tutmak amacıyla batch boyutu artırılmıştır."

---

## Problem 4: Deneysel Aşırı Karmaşıklık

### Problem Nasıl Anlaşılıyor?
Commit `7caadc8` (revert), DAgger, scripted expert, promotion, temporal, effective_loss gibi birden fazla gelişmiş bileşenin tek seferde kaldırılmasını temsil etmektedir. Bu, sistemin yönetilmesi zor bir karmaşıklık düzeyine ulaştığına işaret etmektedir.

### İlgili Commitler
- `7caadc8` — Revert: phase5 adaptive robust baseline

### Uygulanan Çözüm
Tüm gelişmiş bileşenler kaldırılarak daha önce çalışan bir baseline'a (`5850afb`) geri dönüldü. Daha sade bir mimari üzerine devam edildi.

### Çözümün Etkisi
Sistemin bakım kolaylığı arttı ve geliştirme odaklandı.

### Tezde Kullanılabilecek Açıklama
"Geliştirme sürecinin belirli bir aşamasında sistemin karmaşıklığı yönetilebilir sınırları aştığında, istikrarlı bir önceki baseline'a geri dönme kararı alınmıştır. Bu, yazılım geliştirmede 'daha az bazen daha fazladır' ilkesinin pratikte uygulanmasına örnek teşkil etmektedir."

---

## Problem 5: Pist Etiketleme Gerekliliği

### Problem Nasıl Anlaşılıyor?
Hedef nokta yaklaşımı, her pistin geometrik olarak haritalanmasını gerektirmektedir. Her pist için `raw_trace.csv`, `centerline.csv` ve `labels_*.csv` dosyaları üretilmiştir. Bu, standart DonkeyCar veri toplama sürecine ek bir aşama eklemektedir.

### İlgili Commitler
- `5850afb` — Track mapping: 6 pist için harita verileri
- `0f0b351` — New target point models and mapping utilities for road generation

### Uygulanan Çözüm
`build_target_point_labels.py` ve `track_map.py` ile otomatik pist haritalama pipeline'ı oluşturuldu. Ham sürüş izinden (raw trace) merkez hat (centerline) hesaplanması ve lookahead etiketlemesi otomatikleştirildi.

### Çözümün Etkisi
Yeni pist eklemek için bir kez manuel sürüş yapılması yeterli; sonrası otomatik hale geldi.

### Tezde Kullanılabilecek Açıklama
"Target-point etiketleme sürecinde, her pist için önce ham sürüş izi verisi toplanmış; ardından bu izden merkez hat ve lookahead etiketleri otomatik olarak hesaplanmıştır. Bu iki aşamalı süreç, farklı pist geometrilerine kolayca uyarlanabilmektedir."

---

## Problem 6: Repo Klonlanabilirlik Sorunları

### Problem Nasıl Anlaşılıyor?
2026-04-26 tarihinde arka arkaya 6 commit atıldı:
- `4ba0252` — fix: repo klonlanabilirliğini düzelt
- `d483ea6` — fix: update paths in documentation
- `358d90e` — fix: scripts içindeki hardcoded kişisel yolları kaldır
- `7466d2b` — fix: yeni kullanıcıyı çökertecek 4 tutarsızlığı düzelt
- `fec3302` — fix: kalan küçük tutarsızlıkları gider

Betik ve dokümanlarda hardcoded kişisel yolların (`C:\Users\alper\...` gibi) bulunması, reponun yeni bir kullanıcıda doğrudan çalışmamasına neden oluyordu.

### İlgili Commitler
- `4ba0252`, `d483ea6`, `358d90e`, `7466d2b`, `fec3302`

### Uygulanan Çözüm
Hardcoded yollar dinamik (göreceli) yollarla veya kullanıcının yapılandırması gereken değişkenlerle değiştirildi. README ve `simulationconfig.py` güncellendi.

### Çözümün Etkisi
Repo başka bir ortamda klonlanıp çalıştırılabilir hale getirildi.

### Tezde Kullanılabilecek Açıklama
"Proje, yalnızca geliştiricinin kendi ortamında çalışacak şekilde başlayıp zamanla diğer kullanıcıların da kolayca kurulup çalıştırabileceği bir yapıya dönüştürülmüştür. Bu süreçte hardcoded yollar ve kullanıcıya özgü yapılandırmalar genel parametrelerle değiştirilmiştir."

---

## Problem 7: Veri Toplama Sürecinin İzlenmesi

### Problem Nasıl Anlaşılıyor?
Commit `2995dab`: "create scripts for dataset collection and monitoring"
- `collect_massive_sim_dataset.ps1`: Büyük ölçekli veri toplama
- `monitor_massive_sim_dataset.ps1`: Veri toplama sürecini izleme
- `watch_massive_sim_progress.ps1`: İlerleme takibi

Bu betiklerin oluşturulması, büyük miktarda veri toplamanın uzun sürdüğünü ve sürecin izlenmesine ihtiyaç duyulduğunu göstermektedir.

### İlgili Commitler
- `2995dab` — Veri toplama ve izleme betikleri

### Uygulanan Çözüm
PowerShell otomasyon betikleri ile veri toplama süreci otomatikleştirildi ve gerçek zamanlı izleme sağlandı.
