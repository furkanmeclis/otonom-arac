# Otonom Araç Projesi — Forensic Analiz

Bu dosya, `c:/Users/alper/Desktop/otonom-arac` repository'sinin metrik dosyaları, training logları, closed-loop raporları ve commit geçmişi analiz edilerek oluşturulmuştur. Tüm sayısal değerler somut dosya kaynaklarına dayanmaktadır.

---

## 1. DAgger Pipeline — Neden Kaldırıldı?

### Kanıt
- Commit `7caadc8` mesajı: **"revert: roll back to phase5 adaptive robust baseline (5850afb)"**
- Bu commit ile silinen dosyalar:
  - `target_point/dagger.py`
  - `run_target_point_dagger.py`
  - `target_point/scripted_expert.py`
  - `generate_scripted_target_point_data.py`
  - `analyze_scripted_expert_run.py`

### Teknik Gerekçe
Repository içinde DAgger'ın başarısızlığını kanıtlayan ayrı bir performans karşılaştırma logu bulunamadı. Commit mesajı "phase5 adaptive robust baseline" adlı önceki duruma dönüldüğünü belirtmektedir; bu karar DAgger yaklaşımının baseline'ı geçemediğini ima etmektedir. Ancak bu, commit mesajından çıkarılan bir yorumdur; sayısal kanıt yoktur.

---

## 2. Temporal Model — Neden Kaldırıldı?

### Kanıt
- Commit `7caadc8` ile silinen dosya: `target_point/temporal.py`
- Orphaned config dosyası hâlâ mevcut: `configs/model_12_temporal.py`

### Teknik Gerekçe
Repository içinde temporal modelin performans karşılaştırmasını gösteren log veya metrik bulunamadı. DAgger ile aynı revert commit'inde kaldırılmıştır. Sayısal başarısızlık kanıtı yoktur.

---

## 3. Effective Loss — Neden Kaldırıldı?

### Kanıt
- Commit `7caadc8` ile silinen dosya: `target_point/effective_loss.py`

### Teknik Gerekçe
Repository içinde effective loss ile standart MSE'nin karşılaştırmasını gösteren ablation logu bulunamadı. Sayısal kanıt yoktur.

---

## 4. Promotion System — Neden Kaldırıldı?

### Kanıt
- Commit `7caadc8` ile silinen dosya: `target_point/promotion.py`

### Teknik Gerekçe
Repository içinde promotion system'in kullanımını veya başarısızlığını gösteren log bulunamadı. DAgger pipeline'ıyla birlikte kaldırılmıştır.

---

## 5. Adaptive vs Fixed Lookahead — Karşılaştırma Sonuçları

### Kanıt
Kaynak: `data/sim_multitrack/index/lookahead_report_adaptive_v1.json`, `lookahead_report_fixed_1p2m.json`

| Metrik | Adaptive v1 | Fixed 1.2m |
|---|---|---|
| Toplam örnek | 641.612 | 641.625 |
| Ortalama lookahead | **1.035 m** | 1.200 m (sabit) |
| Std dev | 0.173 m | ~0 (5.96e-7) |
| p95 | 1.376 m | 1.200 m |

**Pist bazlı karşılaştırma:**

| Pist | Adaptive Ortalama | Fixed Değer | Fark |
|---|---|---|---|
| circuit-launch | 0.940 m | 1.200 m | -0.260 m |
| generated-roads | 1.068 m | 1.200 m | -0.132 m |
| generated-track | 1.009 m | 1.200 m | -0.191 m |
| minimonaco | 1.040 m | 1.200 m | -0.160 m |
| roboracingleague | 0.986 m | 1.200 m | -0.214 m |
| warehouse | 1.077 m | 1.200 m | -0.123 m |
| warren | 1.121 m | 1.200 m | -0.079 m |

**Teknik gözlem:** Adaptive lookahead, kıvrımlı pistlerde (circuit-launch) daha kısa mesafe (0.940 m) kullanırken düz pistlerde (warren) daha uzun mesafe (1.121 m) kullanmaktadır. Bu davranış beklenen bir adaptasyon göstermektedir.

**Closed-loop karşılaştırma:** Repository içinde adaptive vs fixed'ın doğrudan closed-loop başarı oranı karşılaştırması bulunamadı.

---

## 6. Closed-Loop Başarı Oranları

### Kanıt
Kaynak: `data/artifacts/target_point/reports/closed_loop_20260420T101113Z/closed_loop_summary.json`

**Model: sim_multitrack_v1 — Tüm Pistler:**

| Pist | Episode | Tamamlama % | Başarısızlık Nedeni | Başarısızlık Süresi |
|---|---|---|---|---|
| donkey-generated-track-v0 | 1 | **2.30%** | offtrack | 3.15 s |
| donkey-warren-track-v0 | 1 | **1.62%** | offtrack | 3.20 s |
| donkey-warehouse-v0 | 1 | **0.93%** | offtrack | 2.50 s |
| donkey-minimonaco-track-v0 | 1 | **0.0006%** | offtrack | 0.20 s |
| donkey-circuit-launch-track-v0 | 1 | **0.0016%** | offtrack | 0.20 s |

**Genel istatistikler:**
- Toplam episode: 5
- Başarısızlık: 5 (%100)
- Recovery girişimi: 5
- Recovery başarısı: **0** (%0)
- Geçerli tahmin oranı: %100 (tahminler üretildi, ancak yanlıştı)

---

## 7. Hangi Pistte Hangi Model Başarısız Oldu?

### Kanıt
Kaynak: `data/artifacts/target_point/reports/model_smoke_model_02_sim_domain_randomization/closed_loop_summary.json`

**En Kötü Performans Gösteren Pistler (Tüm Modeller):**

1. **circuit-launch-track-v0** — Kıvrım süresi 0.20 s, center track error 3.95 m, eğrilik 0.56
2. **minimonaco-track-v0** — Kıvrım süresi 0.20 s, center track error 1.94 m, eğrilik 0.56
3. **generated-roads-v0** — Tüm modeller başarısız oldu

**Ortak başarısızlık paterni:** Yüksek eğrilikli kıvrımlar (curvature ≥ 0.56). Model düz yollarda 2.4–3.2 saniye hayatta kalabilmekte, ancak ilk kıvrımda 0.2 saniyede başarısız olmaktadır.

---

## 8. Domain Randomization — Fayda Sağladı mı?

### Kanıt
Kaynak: `data/artifacts/target_point/reports/model_smoke_model_02_sim_domain_randomization/closed_loop_summary.json`, `data/artifacts/target_point/experiments/multi_model_first6/20260421_102008/summary.csv`

**Eğitim karşılaştırması:**

| Model | Eğitim Süresi | Yöntem |
|---|---|---|
| MODEL-01 (pure_sim) | **900 s** | Sadece simülasyon |
| MODEL-02 (domain_randomization) | **16.506 s** | Domain randomization |

**Closed-loop sonuçları:**

| Model | Test Pisti | Tamamlama | Başarısızlık Süresi | Ortalama CTE |
|---|---|---|---|---|
| MODEL-01 (pure_sim) | generated-roads-v0 | %0 | 2.4 s | 0.268 m |
| MODEL-02 (domain_randomization) | generated-roads-v0 | **%0** | 2.5 s | **0.403 m** |

**Teknik sonuç:** Domain randomization 18.3x daha uzun eğitim süresi gerektirdi. Closed-loop başarı oranı değişmedi (ikisi de %0). Domain randomization modeli (MODEL-02) aynı pistte daha yüksek CTE=0.403 m elde etti; baseline MODEL-01 ise CTE=0.268 m ile daha düşük hata gösterdi.

---

## 9. Hangi Model En İyi Sonucu Verdi?

### Kanıt
Kaynak: `data/artifacts/target_point/experiments/multi_model_first6/20260421_102008/summary.csv`

**6 model kapalı döngü karşılaştırması (generated-roads pisti):**

| Model | Eğitim Süresi | Tamamlama | Başarısızlık Süresi | Ortalama CTE |
|---|---|---|---|---|
| MODEL-01 (pure_sim) | 900 s | **%0** | 2.4 s | **0.268 m** |
| MODEL-02 (domain_randomization) | 16.506 s | %0 | 2.5 s | 0.403 m |
| MODEL-03 (pure_real) | 3.061 s | %0 | 1.95 s | 0.323 m |
| MODEL-04 (hybrid_v1_naive_mix) | 3.502 s | %0 | 2.2 s | 0.322 m |
| MODEL-05 (hybrid_v2_sim_heavy) | 14.448 s | %0 | — | — |
| MODEL-06 (hybrid_v3_real_heavy) | 20 s | %0 | — | — |

**Tüm modeller closed-loop testini geçemedi (%0 tamamlama oranı).**

En düşük CTE: MODEL-01 (pure_sim) — 0.268 m. En uzun hayatta kalma süresi: MODEL-02 — 2.5 s (ancak en yüksek CTE ile).

---

## 10. Sim-to-Real Farkı Ölçüldü mü?

### Kanıt
Kaynak: `data/sim_multitrack/index/filter_report.json`

**Eğitim verisi başarı istatistikleri:**

| Kategori | Sayı |
|---|---|
| Toplam warmup frame (yoksayılan) | 22.260 |
| Off-track (sadece nominal) | 135.294 |
| Başarısız recovery girişimi | 26.319 |
| **Başarılı recovery** | **126** |
| Recovery başarı oranı | **%0.48** |

**Teknik gözlem:** Eğitim verisinde dahi recovery başarı oranı %0.48'dir (126 başarı / 26.445 girişim). Closed-loop testinde bu oran %0'a düşmüştür. Bu, sim-to-real farkından ziyade temelden zayıf bir recovery mekanizmasına işaret etmektedir.

Gerçek sim-to-real transfer ölçümü (simülasyon ortamında başarılı → gerçek ortamda test) repository içinde bulunamadı; tüm testler simülasyon ortamında yapılmıştır.

---

## 11. Hangi Committen Sonra Performans Değişti?

### Kanıt
**Büyük revert commit:** `7caadc8` — "revert: roll back to phase5 adaptive robust baseline (5850afb)"

Bu commit şu dosyaları kaldırdı:
- `target_point/dagger.py`
- `target_point/temporal.py`
- `target_point/effective_loss.py`
- `target_point/promotion.py`
- `target_point/scripted_expert.py`
- `generate_scripted_target_point_data.py`
- `run_target_point_dagger.py`
- `run_single_track_stabilization.py`
- `myconfig_phase16_balanced_v3.py`
- `myconfig_phase17_curvature_narrow.py`
- `myconfig_phase19_mid_corner_carry.py`

**Revert öncesi-sonrası performans karşılaştırması:** Repository içinde bu commit öncesi ve sonrasını karşılaştıran closed-loop logu bulunamadı.

---

## 12. Training Loss Seyri

### Kanıt
Kaynak: `data/artifacts/target_point/experiments/multitrack_320x320_efficient_v1/history.csv`

| Epoch | Train Loss | Val Loss | Learning Rate |
|---|---|---|---|
| 0 | 3.704 | 0.4998 | 0.0005 |
| 1 | 3.217 | 0.6015 | 0.0005 |
| 2 | 2.971 | 0.6347 | 0.0005 |
| 3 | 2.883 | 0.5080 | 0.0005 |
| 4 | 2.838 | 0.6088 | 0.00025 |
| 5 | 2.602 | 0.5576 | 0.00025 |
| 6 | 3.927 | 0.6556 | 0.00025 |
| 7 | 2.850 | 0.6543 | 0.000125 |
| 8 | 3.944 | 0.5845 | 0.000125 |

**Gözlem:** Training loss 3.704→2.602 (%30 düşüş), validation loss 0.50–0.66 aralığında sabit (gürültülü). Overfitting veya veri kalitesi sorunu göstergesi.

---

## 13. Veri Filtreleme İstatistikleri

### Kanıt
Kaynak: `data/sim_multitrack/index/filter_report.json`

| Filtre Tipi | Sayı | Açıklama |
|---|---|---|
| Warmup frame | 22.260 | Kalibrasyon frame'leri (yoksayıldı) |
| Off-track (nominal) | 135.294 | Araç pisti terk etti |
| Başarısız recovery | 26.319 | Recovery manevrası başarısız |
| Episode tamamlandı | 401 | Normal sonlanma |
| Yaw jump | 2 | Sensör hatası |
| **Başarılı recovery** | **126** | Recovery başarısı |

**Kritik bulgu:** 26.445 recovery girişiminin yalnızca 126'sı (%0.48) başarılıdır.

---

*Bu dosya, `c:/Users/alper/Desktop/otonom-arac` repository'sinin forensic analizi temelinde 2026-05-16 tarihinde oluşturulmuştur. Tüm sayısal değerler ilgili kaynak dosyalara atıfla verilmiştir.*
