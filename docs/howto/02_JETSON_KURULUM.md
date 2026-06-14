# Jetson Nano Kurulum ve Sürüş Rehberi (Detaylı)

Bu rehber, eğitilmiş modeli **gerçek arabada** çalıştırmak için sıfırdan,
adım adım, her şeyi açıklayarak anlatır. Donanım terimlerini de açıklar.

---

## Önce kafamızdaki resim

Gerçek araba şu parçalardan oluşur:

```
[USB Kamera] → görüntü → [Jetson Nano = küçük bilgisayar] → karar
                                  │
                          [PCA9685 kartı] = Jetson'un "elektrik çevirmeni"
                                  │
                    ┌─────────────┴─────────────┐
              [Direksiyon servosu]         [ESC → Motor]
```

- **Jetson Nano:** Arabanın beyni. Bizim modelimiz burada çalışır.
- **PCA9685:** Jetson dijital komut verir ("%30 sağa kır"), ama servo/motor
  **PWM sinyali** anlar. PCA9685 bu çeviriyi yapan küçük bir karttır. Jetson'a
  **I2C** denen kablo protokolüyle bağlıdır.
- **PWM:** Servo/motora "ne kadar" dönmesini söyleyen elektrik sinyali. Her
  arabada değerleri farklıdır, o yüzden **kalibrasyon** yaparız.
- **Tub:** DonkeyCar'ın veri kaydettiği klasör (görüntüler + komutlar).

> Aşağıdaki komutların hepsi **Jetson üzerinde terminalde** çalışır.

---

## Gereksinimler (kuruluma başlamadan)

- Jetson Nano açık, JetPack + DonkeyCar kurulu
- USB kamera takılı
- PCA9685 kartı I2C ile bağlı, ona direksiyon servosu + ESC bağlı
- Joystick (Xbox/PS) takılı
- Jetson ile aynı Wi-Fi'da bir bilgisayar (uzaktan bakmak için)

---

## AŞAMA 2 — Projeyi indirme ve kurma

**Amaç:** Bizim kodumuzu ve modelleri Jetson'a getirmek.

### 2.1 — Projeyi indir
```bash
git clone <senin-github-repo-linkin>
```
**Ne oldu?** GitHub'daki projenin kopyası Jetson'a indi. İçinde kod + eğitilmiş
modeller var (büyük veri inmez, gerek yok).

### 2.2 — Ayar dosyasını oluştur
```bash
cp .env.example .env
```
**Ne oldu?** `.env` = makineye özel ayarlar dosyası. Gerçek araçta içeriği
varsayılan kalabilir (simülatör yolu gerekmez).

### 2.3 — Gerekli paketleri kur
```bash
pip install -r requirements-train.txt
```
**Ne oldu?** Modelin çalışması için gereken Python kütüphaneleri kuruldu.
(DonkeyCar zaten kuruluydu; bu sadece eksikleri tamamlar.)

---

## AŞAMA 3 — Arabayı tanıtma (`config.py` ayarları)

**Amaç:** `config.py` dosyasına "benim kameram şu, motorum şöyle bağlı" demek.
Dosyayı bir editörle aç (`nano config.py` veya VS Code).

### 3.1 — USB kamerayı tanıt

Önce kameranın numarasını öğren:
```bash
ls /dev/video*
```
**Ne göreceksin?** `/dev/video0` gibi bir çıktı. Bu USB kameranın numarasıdır.
`video0` → numara **0**, `video1` → **1**.

`config.py`'de şu satırları bul ve ayarla:
```python
CAMERA_TYPE = "WEBCAM"     # USB kamera demek
CAMERA_INDEX = 0           # yukarıda gördüğün numara
IMAGE_W = 320              # kameranın çektiği genişlik
IMAGE_H = 240              # kameranın çektiği yükseklik
```
**Neden?** "WEBCAM" modele USB kamera kullanacağını söyler. `IMAGE_W/H` kameranın
çektiği boyut — model bunu otomatik 128×128'e küçültür, sen küçültmeyi düşünme.

### 3.2 — Motor kartını (PCA9685) tanıt

Kartın "adresini" öğren:
```bash
sudo i2cdetect -y -r 1
```
**Ne göreceksin?** Bir tablo. İçinde **40** yazan bir hücre olmalı — bu PCA9685'in
I2C adresidir. (Görünmüyorsa kart bağlı değil / kablo gevşek demektir.)

`config.py`'de:
```python
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"  # PCA9685 ile sürüş (zaten ayarlı)
PCA9685_I2C_ADDR = 0x40                      # i2cdetect'te gördüğün 40
PCA9685_I2C_BUSNUM = 1                        # Jetson'da genelde 1
```

### 3.3 — Direksiyon ve gazı kalibre et (EN ÖNEMLİ ADIM)

**Neden gerekli?** Her servo/motor farklı PWM değeriyle çalışır. "Tam sol" için
hangi sayı lazım, arabaya sorarak buluruz.

**Direksiyon:**
```bash
donkey calibrate --channel 1 --bus 1
```
**Ne olacak?** Ekran senden sayı isteyecek (örn. `360`). Yazıp Enter'a basınca
**tekerlek o yöne döner.** Farklı sayılar dene:
- Tekerlek **tam sola** dönene kadar dene → o sayı **STEERING_LEFT_PWM**
- Tekerlek **tam sağa** dönene kadar dene → o sayı **STEERING_RIGHT_PWM**

**Gaz:**
```bash
donkey calibrate --channel 0 --bus 1
```
- Araba yavaşça **ileri** gitmeye başladığı sayı → **THROTTLE_FORWARD_PWM**
- Araba **durduğu** (nötr) sayı → **THROTTLE_STOPPED_PWM**

> ⚠️ Gaz kalibre ederken arabayı **askıya al** (tekerlekler yere değmesin) ki
> kaçmasın.

Bulduğun sayıları `config.py`'ye yaz:
```python
STEERING_LEFT_PWM = 460       # senin "tam sol" değerin
STEERING_RIGHT_PWM = 290      # senin "tam sağ" değerin
THROTTLE_FORWARD_PWM = 500    # ileri
THROTTLE_STOPPED_PWM = 370    # nötr/durma
```

### 3.4 — Model ayarları (zaten doğru, sadece kontrol et)

`config.py`'nin sonunda bizim eklediğimiz blok var:
```python
TARGET_POINT_IMAGE_W = 128        # model 128 ile eğitildi — DEĞİŞTİRME
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_INVERT_COLORS = False
TARGET_POINT_BASE_THROTTLE = 0.18 # ilk test için düşük hız (güvenli)
```
**Neden 128 değişmez?** Model eğitimde 128×128 görüntü gördü. Farklı verirsen
model şaşırır. Bu yüzden sabit.

---

## AŞAMA 4 — İlk test: araba kendi sürsün

**Hangi modeli kullanacaksın?** Pistinin rengine göre seç:

| Pistin böyleyse | Bu modeli kullan |
|---|---|
| Koyu zemin + beyaz bant | `model_03_pure_real_fp16.tflite` |
| Açık zemin + kırmızı/beyaz bant | `model_ucsd_fp16.tflite` |
| Beyaz zemin + siyah bant | `model_01_inverted_fp16.tflite` |

Sürüş komutu:
```bash
python manage.py drive --model models/tflite/model_03_pure_real_fp16.tflite --type target_point --simulationconfig=config.py
```
**Ne oldu?** Araba modeli yükledi ve **kendi kendine sürmeye** başlar.

**Sürerken:**
- Bilgisayarında tarayıcı aç: `http://<jetson-ip>:8887` → arabanın gördüğünü
  canlı izlersin. (IP için Jetson'da: `hostname -I`)
- ⚠️ **İlk denemede gazı çok düşük tut**, joystick elinde olsun. Kaçarsa hemen müdahale et.

**İyi sürüyorsa** → tuttu. **Kötü sürüyorsa** → normal, Aşama 5.

---

## AŞAMA 5 — Model kötü sürerse: kendi pistinden veri topla

Model gerçek pistinde tökezlerse (sık olur, çünkü senin pistin modelin gördüğünden
farklı), **kendi pistinden örnek toplarsın.**

```bash
python manage.py drive --js
```
**Ne oldu?** Model YOK — **sen** joystick'le sürersin. Gaz verince DonkeyCar
otomatik kaydeder (her an: görüntü + senin direksiyon + gaz).

**Nasıl sürmelisin? (etiket kaliten = sürüş kaliten)**
- **Temiz turlar** at (pisti düzgün takip et)
- Arada **kenara yaklaşıp merkeze dön** (kurtarma — hatadan dönmeyi öğretir)
- **İki yönde** sür
- Sürüş yapacağın **ışıkta** topla
- Toplam **10-15 dakika** yeter

**Kayıt oluyor mu?** Terminalde `recorded 100 records` mesajları görürsün.
Kayıtlar `data/` altına gider (`ls data/`).

**Sonra:**
```
1. data/ klasörünü SENİN bilgisayarına yolla (USB/bulut)
2. Sen GPU'nda fine-tune edersin (kısa eğitim)
3. Yeni modeli Jetson'a geri yollarsın
4. Tekrar test (Aşama 4)
```
> "Etiketleme" diye ayrı bir adım YOK — gerçek veride etiket senin direksiyonundan
> otomatik üretilir.

---

## Faydalı komutlar — hangisi ne öğretir?

| Komut | Ne öğrenirsin / yapar |
|---|---|
| `ls /dev/video*` | USB kamera numarası → `CAMERA_INDEX` |
| `sudo i2cdetect -y -r 1` | PCA9685 adresi (genelde 40) |
| `donkey calibrate --channel 1 --bus 1` | Direksiyon PWM değerleri |
| `donkey calibrate --channel 0 --bus 1` | Gaz PWM değerleri |
| `hostname -I` | Jetson'un IP'si (web arayüzü için) |
| `python manage.py drive --js` | Manuel sür + veri kaydet |
| `ls data/` | Toplanan tub klasörleri |
| `nano config.py` | Ayar dosyasını düzenle |

---

## Sorun giderme

| Belirti | Sebep / Çözüm |
|---|---|
| "Kamera bulunamadı" | `ls /dev/video*` boş → kamera takılı değil. Numara yanlışsa `CAMERA_INDEX` düzelt. |
| `i2cdetect`'te 40 yok | PCA9685 bağlı değil / kablo gevşek / güç yok |
| Direksiyon **ters** kırıyor | `config.py`: `TARGET_POINT_STEER_SIGN = -1.0` |
| Sürekli **bir yana** çekiyor | `TARGET_POINT_TARGET_X_BIAS_M` ayarla |
| Çok **hızlı** | `TARGET_POINT_BASE_THROTTLE` düşür (örn. 0.15) |
| **Virajı kaçırıyor** | `TARGET_POINT_STEER_GAIN` artır (örn. 1.6) |
| **Zikzak** yapıyor | `TARGET_POINT_STEER_GAIN` azalt |
| Web arayüzü açılmıyor | `hostname -I` ile IP doğru mu, aynı Wi-Fi mi |
| `import tensorflow` hatası | Jetson'da TF yok → `tflite_runtime` kurulup kod uyarlanmalı (söyle, yaparım) |

> 💡 Bu tablodaki `TARGET_POINT_*` ayarları **modelde değil kontrolcüde** —
> değiştirmek için **yeniden eğitim gerekmez**, sadece `config.py`'yi düzenleyip
> tekrar sür.

---

## Tek bakışta tüm akış

```
2. İndir + kur        (git clone, pip install)
3. Arabayı tanıt      (kamera + PCA9685 + kalibrasyon)
4. Test et            (drive --model ...)  → iyiyse bitti
5. Olmazsa veri topla (drive --js) → sana yolla → fine-tune → tekrar
```

İlgili: [CONFIG_REHBERI.md](04_CONFIG_REHBERI.md) · [MODEL_DEGERLENDIRME.md](01_MODEL_DEGERLENDIRME.md)
