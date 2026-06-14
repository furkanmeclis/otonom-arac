# Kurulum Rehberi

## Gereksinimler

- Python **3.11** (3.12 desteklenmiyor)
- Windows 10/11
- DonkeySim binary (aşağıya bakın)

---

## 1. DonkeySim İndir

Simülatörün Windows binary'sini aşağıdaki adresten indir:

```
https://github.com/tawnkramer/gym-donkeycar/releases
```

`DonkeySimWin.zip` dosyasını indir, istediğin bir yere çıkart. İçindeki `donkey_sim.exe` dosyasının yolunu bir sonraki adımda kullanacaksın.

---

## 2. Repoyu İndir

```powershell
git clone <repo-url>
cd otonom-arac\otonom-arac
```

---

## 3. Sanal Ortam Oluştur

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements-train.txt
```

> Tüm komutlar `.venv\Scripts\python` ile çalıştırılmalıdır.

---

## 4. Konfigürasyon

`simulationconfig.py` içinde aşağıdaki satırı kendi DonkeySim yoluna göre düzenle:

```python
# Örnek:
DONKEY_SIM_PATH = "C:/your/path/to/DonkeySimWin/donkey_sim.exe"

# Simülatörü elle başlatmak istersen:
# DONKEY_SIM_PATH = "remote"
```

---

## 5. Kurulumu Doğrula

Simülatörü açıp kamera ve telemetriyi test etmek için:

```powershell
.\.venv\Scripts\python manage.py smoke --simulationconfig=simulationconfig.py
```

Beklenen çıktı sonunda `[smoke] success` görünmelidir.

---

## 6. Çalışmaya Hazır Hale Getirme

Eğer mevcut eğitilmiş model kullanılacaksa kurulum bu kadar.  
Sıfırdan veri toplayıp model eğitmek için `COMMANDS.md` dosyasına bak.

---

## Mevcut Eğitilmiş Model

```
models/target_point_combined_large_noaug.keras
```

Test etmek için:

```powershell
.\.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-generated-roads-v0 `
  --episodes-per-track 10
```

---

## Önemli Notlar

- Python 3.12 **desteklenmiyor** — sadece **3.11** kullan
- `tensorflow==2.15.1` ve `keras==2.15` gereklidir (requirements-train.txt içinde)
- Simülatör otomatik başlatılır, ayrıca elle açmana gerek yok
- Tüm komutlar `otonom-arac\otonom-arac\` dizininden çalıştırılır
