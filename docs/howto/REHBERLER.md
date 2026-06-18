# Proje Rehberleri — Çalışma Sırası

Bu projeye özel Türkçe dokümanlar. **Numaralar önerilen okuma/çalışma sırasıdır.**

| Sıra | Doküman | Ne zaman okursun |
|---|---|---|
| **00** | [00_PIPELINE_REHBER.md](00_PIPELINE_REHBER.md) | **Önce bunu** — projenin genel akışı: veri → eğitim → sürüş. Hangi dosya ne yapar. |
| **01** | [01_MODEL_DEGERLENDIRME.md](01_MODEL_DEGERLENDIRME.md) | Elimizdeki modeller; pistine göre hangisini seçersin; Jetson-hazır tflite'lar. |
| **02** | [02_JETSON_KURULUM.md](02_JETSON_KURULUM.md) | Gerçek arabayı (Jetson + USB kamera) kur, kalibre et, modeli **test et**. |
| **03** | [03_KENDI_MODELINI_EGIT.md](03_KENDI_MODELINI_EGIT.md) | Model iyi süremezse: kendi verinle **fine-tune** + TFLite'a çevir. |
| **04** | [04_CONFIG_REHBERI.md](04_CONFIG_REHBERI.md) | Başvuru: `config.py`'deki her ayar ne işe yarar. |
| **05** | [05_DONKEYCTL_LAUNCHER.md](05_DONKEYCTL_LAUNCHER.md) | `donkeyctl.py` ile sim'i tek komutla başlat, tünel aç, telefonu QR ile bağla. |

## Tipik çalışma akışı

```
00 (anla)  →  01 (model seç)  →  02 (Jetson'da test et)
                                      │
                              iyi mi? ─┤
                                      │ hayır
                                      ▼
                              03 (kendi verinle eğit) → tekrar 02
                              
04 = her aşamada başvuru (config ayarları)
```

> Not: `docs/` altındaki diğer dosyalar (COMMANDS.md, SETUP.md, REAL_TRACK_PREP.md)
> DonkeyCar'ın kendi dokümanlarıdır.
