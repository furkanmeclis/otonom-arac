# Otonom Araç — Target-Point Sim-to-Real

DonkeyCar tabanlı otonom araç projesi. Model, kameradan doğrudan direksiyon açısı yerine
pistin ilerleyen bir noktasının `(target_x, target_y)` koordinatını tahmin eder.
Bu yaklaşım sim-to-real transferini kolaylaştırır.

## Hızlı Başlangıç

Kurulum, veri toplama, eğitim ve sürüş adımları için:

```
CALISTIRMA_REHBERI.md
```

Projenin tüm dosyalarının açıklaması için:

```
PROJE_ACIKLAMA.md
```

Model eğitim sürecinin teknik detayları için:

```
MODEL_EGITIM_SURECI.md
```

## Gereksinimler

- Python 3.11 (`tensorflow==2.15.1` yalnızca 3.11 ile çalışır)
- [DonkeySim binary](https://github.com/tawnkramer/gym-donkeycar/releases) — `DonkeySimWin.zip`
- NVIDIA GPU (opsiyonel, CPU ile de eğitim yapılabilir)

## Kurulum

```powershell
git clone <repo-url>
cd otonom-arac\otonom-arac

py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-train.txt
```

`simulationconfig.py` içindeki `DONKEY_SIM_PATH` değerini kendi DonkeySim yoluna güncelle, ardından:

```powershell
.\.venv\Scripts\python manage.py smoke --simulationconfig=simulationconfig.py
```

## Mimari Özeti

| Bileşen | Açıklama |
|---------|----------|
| **Model** | Efficient CNN (~115K parametre, depthwise separable conv) |
| **Çıktı** | `(target_x, target_y)` — ego-frame'de metre cinsinden |
| **Controller** | Geometrik; heading error → direksiyon, viraj açısına göre gaz |
| **Veri Formatı** | JSONL manifest + DonkeyCar Tub V2 görüntüleri |
| **Eğitim** | TensorFlow 2.15.1 / Keras, sample weighting, temporal augmentation |

## Proje Yapısı

```
otonom-arac/
├── ai_pipeline/
│   ├── target_point/         # model, controller, dataset, augment
│   ├── train.py              # eğitim giriş noktası
│   ├── collect_target_point_data.py
│   ├── build_target_point_labels.py
│   └── evaluate_target_point.py
├── configs/                  # model_01 … model_12 deneyleri
├── data/
│   ├── sim_unified_maps/     # boş dizin yapısı (.gitkeep)
│   └── artifacts/maps/       # pist merkez hattı haritaları (--task map sonrası oluşur)
├── models/                   # eğitilmiş .keras dosyaları
├── manage.py                 # araç sürüş / smoke test
├── simulationconfig.py       # tüm parametreler burada
└── requirements-train.txt
```

## Lisans

MIT
