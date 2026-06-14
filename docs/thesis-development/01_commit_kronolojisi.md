# 01 — Commit Kronolojisi

Commitler en eskiden en yeniye doğru sıralanmıştır. Her commit için teknik içerik analiz edilmiştir. Commit mesajları zaman zaman yetersiz olduğundan yorum, diff içeriğine dayandırılmıştır.

---

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

---

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

---

## Önemli Notlar

- Commit `2952627` ("last") ve `41389c3` ("Son yapılan alan") anlamsız mesaj içermektedir. Diff incelemesi yapılmadan teknik içerikleri belirlenemez.
- Commit `7caadc8` (revert) projedeki en büyük geri alma olayıdır; 165 dosya etkilenmiş ve önemli deneme modülleri kaldırılmıştır.
- 2026-04-26 tarihindeki 6 commit kümesi, proje sonlanmadan önce yayına hazırlık sürecine işaret etmektedir.
