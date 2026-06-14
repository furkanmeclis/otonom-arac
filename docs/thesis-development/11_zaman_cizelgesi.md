# 11 — Zaman Çizelgesi

Commit tarihleri kullanılarak projenin kronolojik gelişimi aşağıda tablolaştırılmıştır.

---

## Zaman Çizelgesi Tablosu

| Tarih | Geliştirme Aşaması | İlgili Commitler | Açıklama |
|-------|-------------------|-----------------|----------|
| 2026-03-05 | Proje Kurulumu | `eee7f31`, `c5b68e3` | DonkeyCar çerçevesi kuruldu. İlk eğitim betiği ve cihaz yapılandırması eklendi. |
| 2026-03-11 | İlk Model Konfigürasyonu | `b28d47b` | `lane_v1` model konfigürasyonu ve myconfig güncellemesi. |
| 2026-03-24 | Target-Point Modülü | `6a8538a` | Ana özellik: `target_point/` modülü (model, kontrolcü, dataset, eğitim, pilot, testler). |
| 2026-03-27 | Pist Haritalama + Phase 4-5 | `5850afb` | 6 pist haritalandı. Phase 4 ve Phase 5 deneyleri (adaptive/fixed). 43K+ satır ekleme. |
| 2026-03-29 | Phase 5.5 Değerlendirme | `e02951f`, `ac4920b` | Phase 5.5 sabit bootstrap raporları ve yeni eğitim bileşenleri. |
| 2026-03-31 | Tek-Pist Stabilizasyon Denemesi | `10cf4b9` | Tek-pist stabilizasyon betiği (sonradan kaldırıldı). |
| 2026-04-01 | Temizleme | `2952627`, `e9f3908` | Küçük değişiklik ("last") ve Unity log temizliği. |
| 2026-04-03 | Büyük Revert | `7caadc8`, `31f73db`, `2e9d6e4` | DAgger, scripted expert, temporal, promotion, effective_loss kaldırıldı. Phase 5 adaptive robust baseline'a dönüş. |
| 2026-04-05 | Yeni Pist Denemeleri | `0f0b351`, `8231732`, `41389c3` | Yol üretimi için yeni modeller ve haritalar. 5 koşu özet raporu. |
| 2026-04-06 | Kapalı Döngü Değerlendirme | `a4c40f7`, `fcb6f8f`, `233ad5a` | Birden fazla sürüş oturumu için kapalı döngü raporları (3 commit). |
| 2026-04-07 | Dokümantasyon Yazma | `43894f0`, `4f802fc`, `7236feb` | COMMANDS.md, SETUP.md ve PowerShell dokümantasyonu. |
| 2026-04-08 | Unity Log | `2c25c81` | Unity performans logu eklendi. |
| 2026-04-16 | Sim2Real Kılavuzları | `c8e688f`, `c98a637` | Çok pist sim2real kılavuzu ve 3 tur koşu betiği. |
| 2026-04-17 | Büyük Mimari Yeniden Yapılanma | `5de581c`, `40d035d`, `95dafd8` | `ai_pipeline/` yapısına geçiş. Supervised learning örnekleri kaldırıldı. TFLite desteği eklendi. |
| 2026-04-18 | Unity Log | `6687c15` | Yeni Unity debug logu. |
| 2026-04-19 | Kontrolcü İyileştirme + Veri Toplama | `2995dab` | Bias kompansasyonu, deadband. Büyük ölçekli veri toplama betikleri. |
| 2026-04-21 | Çok Model Eğitim Planı | `ec10b31` | MODEL_TRAINING_PLAN.md. Model 01-06 konfigürasyonları. Batch size 32→128. |
| 2026-04-22 | Ek Model Konfigürasyonları | `3f66a3f` | Model 07 (fine-tune), 11 (multi-task), 12 (temporal). model_export_manifest.json. |
| 2026-04-26 | Yayına Hazırlık (6 commit) | `0ce7fd9` → `fec3302` | Kapsamlı dokümantasyon. Hardcoded yollar kaldırıldı. Repo klonlanabilirliği sağlandı. |

---

## Gelişim Yoğunluğu Analizi

### Commit Dağılımı (Tarih Bazında)

| Ay | Commit Sayısı | Yoğunluk |
|----|--------------|----------|
| Mart 2026 (5–31) | 8 commit | Temel altyapı ve ana özellik |
| Nisan 2026 (1–15) | 16 commit | Deneyler, revert, değerlendirme |
| Nisan 2026 (16–22) | 8 commit | Yeniden yapılanma ve çok model |
| Nisan 2026 (26) | 7 commit | Yayına hazırlık |

### Önemli Tarihler

| Tarih | Olay |
|-------|------|
| 2026-03-05 | Proje başlangıcı |
| 2026-03-24 | Target-point modülünün ilk entegrasyonu |
| 2026-03-27 | En büyük tek commit (43K+ satır, pist haritalama + phase 4-5) |
| 2026-04-03 | En büyük revert (165 dosya, 4829 satır silindi) |
| 2026-04-17 | Mimari yeniden yapılanma (`ai_pipeline/`) |
| 2026-04-26 | Son commit (yayına hazırlık tamamlandı) |

---

## Geliştirme Süresi

İlk commit (2026-03-05) ile son commit (2026-04-26) arasında geçen süre: **52 gün**

Bu süre içinde toplam **39 commit** atılmıştır.

Ortalama: ~0.75 commit/gün

Commit yoğunluğu Nisan başında ve Nisan sonunda zirveye ulaşmıştır:
- Nisan 3: 3 commit (revert ve temizleme)
- Nisan 26: 7 commit (yayına hazırlık)

---

## Zaman Çizelgesi Grafiği (Metin Tabanlı)

```
MART 2026
Hf1 ████░░░░  [05 Mart] Proje kurulumu
Hf2 ░░░░░░░░
Hf3 ░░████░░  [11 Mart] lane_v1 config
Hf4 ░░░░░░░░
Hf5 ░░░░████  [24 Mart] target_point/ modülü
               [27 Mart] Pist haritalama + Phase 4-5

NİSAN 2026
Hf1 ████████  [29-31 Mart / 01-03 Nisan] Değerlendirme + REVERT
Hf2 ████████  [05-08 Nisan] Yeni denemeler + Dokümantasyon
Hf3 ░░░░████  [16-17 Nisan] Sim2Real + Mimari yeniden yapılanma
Hf4 ████░░░░  [18-22 Nisan] Kontrolcü + Çok model
Hf5 ████░░░░  [26 Nisan] Yayına hazırlık
```
