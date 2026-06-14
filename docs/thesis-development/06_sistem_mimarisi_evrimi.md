# 06 — Sistem Mimarisi Evrimi

Bu dosyada, projenin başından sonuna kadar yazılım mimarisinin nasıl değiştiği belgelenmektedir.

---

## İlk Mimari Yapı (2026-03-05)

`eee7f31` commitinden sonraki dosya yapısı:

```
otonom-arac/
├── calibrate.py
├── config.py              (770 satır — DonkeyCar config)
├── myconfig.py            (kullanıcı yapılandırmaları)
├── manage.py              (araç yönetimi)
├── train.py               (eğitim giriş noktası)
├── data/                  (veri kümesi — tub formatı)
├── models/
│   ├── dilara.h5
│   ├── my_first_pilot.h5
│   ├── my_2_first_pilot.h5
│   └── database.json
├── docs/
├── examples/
│   ├── genetic_alg/
│   ├── reinforcement_learning/
│   └── supervised_learning/
├── gym_donkeycar/         (DonkeyCar çerçevesi)
├── donkey_sim.app         (macOS simülatör)
└── tests/
```

Bu aşamada sistem, standart DonkeyCar çerçevesinden ibarettir. Model: görüntüden doğrudan direksiyon açısı tahmini.

---

## Ara Mimari 1: Target-Point Modülü (2026-03-24)

`6a8538a` commitinden sonraki eklentiler:

```
otonom-arac/
├── target_point/               ← YENİ MODÜL
│   ├── __init__.py
│   ├── model.py               (CNN, ~ilk versiyon)
│   ├── controller.py          (geometrik kontrolcü)
│   ├── dataset.py             (veri yükleme)
│   ├── training.py            (eğitim döngüsü)
│   ├── diagnostics.py         (teşhis araçları)
│   └── pilot.py               (gerçek zamanlı çıkarım)
├── tests/
│   └── test_target_point.py   ← YENİ
└── ... (önceki dosyalar korundu)
```

### Mimari Değerlendirme
`target_point/` modülü, mevcut DonkeyCar altyapısının yanına ek bir modül olarak eklendi. `manage.py` bu modülü çağıracak şekilde güncellendi. Bu, eklentili (additive) bir mimari evrimdir.

---

## Ara Mimari 2: Pist Haritalama ve Gelişmiş Bileşenler (2026-03-27)

`5850afb` commitinde büyük genişleme:

```
otonom-arac/
├── target_point/
│   ├── ... (önceki dosyalar)
│   ├── augment.py             ← YENİ (veri artırma)
│   ├── collector.py           ← YENİ (veri toplama)
│   ├── domain_randomization.py ← YENİ
│   ├── evaluate_closed_loop.py ← YENİ
│   ├── experiments.py         ← YENİ
│   ├── manifest.py            ← YENİ (JSONL manifest)
│   ├── model.py               (genişletildi)
│   ├── sim_session.py         ← YENİ
│   ├── teacher_policy.py      ← YENİ
│   ├── track_map.py           ← YENİ (pist haritalama)
│   ├── training.py            (büyük ölçüde yeniden yazıldı)
│   ├── dagger.py              ← YENİ (sonradan silindi)
│   ├── scripted_expert.py     ← YENİ (sonradan silindi)
│   ├── promotion.py           ← YENİ (sonradan silindi)
│   ├── temporal.py            ← YENİ (sonradan silindi)
│   └── effective_loss.py      ← YENİ (sonradan silindi)
├── build_target_point_labels.py ← YENİ
├── collect_target_point_data.py ← YENİ
├── evaluate_target_point.py   ← YENİ
└── data/
    └── artifacts/
        └── maps/              ← YENİ (pist haritaları)
```

Bu aşamada `target_point/` modülü en geniş halini aldı. Özellikle `teacher_policy.py` (1003 satır) ve `collector.py` (954 satır) büyük bileşenlerdir.

---

## Mimari Geri Alma (2026-04-03)

`7caadc8` revert ile aşağıdaki bileşenler kaldırıldı:

```
KALDIRILDI:
├── target_point/dagger.py
├── target_point/scripted_expert.py
├── target_point/promotion.py
├── target_point/temporal.py
├── target_point/effective_loss.py
├── run_target_point_dagger.py
├── run_single_track_stabilization.py
├── generate_scripted_target_point_data.py
├── analyze_scripted_expert_run.py
└── myconfig_phase*.py (varyant dosyaları)
```

Ayrıca `controller.py`, `training.py`, `tests/test_target_point.py` ve `manifest.py` büyük ölçüde sadeleştirildi.

### Mimari Değerlendirme
Bu, **subtractive (çıkarıcı) bir mimari adımdır**. Sistem, en karmaşık noktasından geri çekildi. Kalan bileşenler daha sade ve odaklı bir pipeline oluşturmaktadır.

---

## Mimari Yeniden Yapılanma (2026-04-17)

`5de581c` commitinde dizin yapısı yeniden organize edildi:

```
otonom-arac/
├── ai_pipeline/               ← YENİ ÜST DİZİN
│   ├── target_point/          ← TAŞINDI (önceki: target_point/)
│   │   ├── model.py
│   │   ├── controller.py
│   │   ├── dataset.py
│   │   ├── training.py
│   │   ├── augment.py
│   │   ├── collector.py
│   │   ├── domain_randomization.py
│   │   ├── evaluate_closed_loop.py
│   │   ├── experiments.py
│   │   ├── manifest.py
│   │   ├── mapping.py
│   │   ├── model.py
│   │   ├── pilot.py
│   │   ├── pilot_tflite.py    ← YENİ
│   │   ├── export.py          ← YENİ
│   │   ├── external_adapter.py ← YENİ
│   │   ├── sim_session.py
│   │   ├── teacher_policy.py
│   │   └── track_map.py
│   ├── build_target_point_labels.py  ← TAŞINDI
│   ├── collect_target_point_data.py  ← TAŞINDI
│   ├── evaluate_target_point.py      ← TAŞINDI
│   ├── train.py               ← YENİ (ai_pipeline içinde)
│   └── tools/
├── configs/                   ← YENİ (model konfigürasyonları)
│   ├── model_01_pure_sim.py
│   ├── model_02_sim_domain_randomization.py
│   ├── model_03_pure_real.py
│   ├── model_04_hybrid_v1_naive_mix.py
│   ├── model_05_hybrid_v2_sim_heavy.py
│   ├── model_06_hybrid_v3_real_heavy.py
│   ├── model_07_finetune.py
│   ├── model_11_multitask.py
│   └── model_12_temporal.py
├── scripts/                   ← YENİ (otomasyon betikleri)
│   ├── train_first6_models.ps1
│   ├── train_models_5_6.ps1
│   ├── train_models_11_12_07.ps1
│   ├── train_model_01.ps1
│   ├── train_gpu.ps1
│   ├── collect_massive_sim_dataset.ps1
│   ├── monitor_massive_sim_dataset.ps1
│   ├── watch_massive_sim_progress.ps1
│   └── test_model_sim.py
├── examples/                  ← supervised_learning/ kaldırıldı
│   ├── genetic_alg/
│   └── reinforcement_learning/
└── ... (manage.py, simulationconfig.py vb.)
```

---

## Son Mimari Yapı (2026-04-26)

Final commit grubundan sonraki yapı README.md'de özetlenmektedir:

```
otonom-arac/
├── ai_pipeline/
│   ├── target_point/          (ana ML bileşenleri)
│   ├── train.py               (eğitim giriş noktası)
│   ├── collect_target_point_data.py
│   ├── build_target_point_labels.py
│   └── evaluate_target_point.py
├── configs/                   (9 model konfigürasyonu)
├── data/
│   ├── sim_unified_maps/
│   └── artifacts/maps/        (pist haritaları)
├── models/                    (eğitilmiş modeller)
├── scripts/                   (otomasyon betikleri)
├── manage.py
├── simulationconfig.py
└── requirements-train.txt
```

---

## Mimari Değişikliklerin Özeti

| Dönem | Mimari Durum |
|-------|--------------|
| Mart başı | DonkeyCar çerçevesi (düz direksiyon tahmiri) |
| Mart ortası | `target_point/` modülü eklendi |
| Mart sonu | `target_point/` büyük ölçüde genişletildi (DAgger, scripted expert vb.) |
| Nisan başı | Revert: sistem sadeleştirildi |
| Nisan ortası | `ai_pipeline/` yapısına geçiş, `configs/`, `scripts/` eklendi |
| Nisan sonu | Dokümantasyon ve klonlanabilirlik düzeltmeleri |

---

## Mimari Değişikliklerin Yorumu

**Genel eğilim:** Sistem önce genişledi (additive phase), ardından karmaşıklık nedeniyle sadeleştirildi (subtractive phase), sonra yeniden organize edildi (structural reorganization).

Bu tip bir mimari evrimi, araştırma odaklı proje geliştirme sürecinde normaldir: önce birden fazla yaklaşım denenir, ardından en verimli ve yönetilebilir olanlar seçilerek sistem olgunlaştırılır.

### Tezde Kullanılabilecek Anlatım
"Sistem mimarisi, geliştirme sürecinde üç ana dönüşüm geçirmiştir: ilk aşamada DonkeyCar çerçevesine hedef nokta modülü eklenerek genişletilmiş; ardından çeşitli gelişmiş teknikler denenip kaldırılarak sadeleştirilmiş; son aşamada ise çok model denemelerini desteklemek amacıyla `ai_pipeline/` ve `configs/` gibi ayrışık dizin yapılarına geçilmiştir."
