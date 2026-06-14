# config.py Rehberi — Her Ayar Ne İşe Yarar? (Detaylı)

## config.py nedir?

`config.py`, arabanın **tüm ayarlarının** tutulduğu dosyadır: kamera tipi, motor
bağlantısı, hız sınırları, modelin nasıl süreceği... Araba her açıldığında bu
dosyayı okur ve "kendini ona göre kurar".

**Neden bu kadar uzun (800+ satır)?** Çünkü bu, DonkeyCar'ın **standart şablonu.**
DonkeyCar onlarca farklı donanımı destekler (farklı motor sürücüleri, sensörler,
ekranlar...). Dosyada hepsinin ayarı var — ama **sen sadece kendi donanımının
ayarlarını** doldurursun, gerisi olduğu gibi kalır. Yani **çoğu satıra dokunmazsın.**

Buna ek olarak, dosyanın sonunda **bizim eklediğimiz** `TARGET_POINT_*` ayarları
var (modelin nasıl süreceği). Asıl bizi ilgilendiren kısım orası + birkaç donanım
ayarı.

---

## config nasıl yüklenir? (kafa karıştıran kısım)

Bir komut çalıştırdığında ayarlar **iki katmanda** yüklenir:

```
1. config.py        → HER ZAMAN okunur (temel ayarlar)
2. .env             → makineye özel yollar (örn. simülatör yolu) buraya yazılır,
                      config'in üstüne biner
3. --simulationconfig=<dosya>  → varsa, en üste biner (override)
```

**Pratikte:**
- **Gerçek araçta:** komuta `--simulationconfig=config.py` eklersin → sadece
  config.py yüklenir (gerçek donanım modu).
- **Simülasyonda:** `--simulationconfig=simulationconfig.py` → config.py temel +
  simulationconfig.py sim ayarlarını ekler.

> Yani config.py her zaman okunur; üstüne hangi dosyayı bindireceğin moda göre değişir.

---

# 1) BİZİM İÇİN ÖNEMLİ AYARLAR

Bunları gerçekten ayarlarsın/anlarsın.

## PATHS — Klasör yolları
Projedeki klasörlerin yerleri. **Otomatik hesaplanır, dokunmazsın.**

| Ayar | Ne yapar |
|---|---|
| `CAR_PATH` | Projenin kök klasörü (kendiliğinden bulunur) |
| `DATA_PATH` | Toplanan verinin (tub) kaydedileceği yer = `data/` |
| `MODELS_PATH` | Model klasörü |

## VEHICLE — Araç döngüsü
Araba saniyede kaç kez "bak ve karar ver" yapacağı.

| Ayar | Varsayılan | Ne yapar |
|---|---|---|
| `DRIVE_LOOP_HZ` | 20 | Saniyede 20 kez gör→karar. Yükseltirsen daha sık karar ama daha çok işlemci yükü. **20 iyi.** |
| `MAX_LOOPS` | None | Sadece test için döngüyü sınırlama. Normalde None. |

## CAMERA — Kamera (USB)
Modelin "gözü". Yanlış ayarlanırsa görüntü gelmez/bozuk gelir.

| Ayar | Değer | Ne yapar |
|---|---|---|
| `CAMERA_TYPE` | **"WEBCAM"** | Kamera türü. USB için WEBCAM. (PICAM=Pi kamera, CSIC=Jetson CSI) |
| `CAMERA_INDEX` | 0 | Birden fazla kamera varsa hangisi (`ls /dev/video*` ile öğren) |
| `IMAGE_W`/`IMAGE_H` | 320/240 | Kameranın **çektiği** boyut. Model bunu 128'e küçültür; karıştırma. |
| `IMAGE_DEPTH` | 3 | Renkli=3, gri=1 |
| `CAMERA_VFLIP`/`HFLIP` | False | Kamera ters monteliyse görüntüyü çevirir |

**Neden önemli:** Kamera açısı/tipi eğitimdekine benzemezse model şaşırır.

## DRIVE_TRAIN — Motoru nasıl sürüyoruz
Jetson'un dijital komutunu motora ileten **kart (PCA9685)** ayarları.

| Ayar | Değer | Ne yapar |
|---|---|---|
| `DRIVE_TRAIN_TYPE` | **"PWM_STEERING_THROTTLE"** | "PCA9685 kartıyla sürüyorum" demek. Bizimki bu. |
| `PCA9685_I2C_ADDR` | 0x40 | Kartın adresi (`i2cdetect` ile öğren) |
| `PCA9685_I2C_BUSNUM` | 1 | I2C hattı (Jetson'da 1) |
| `PWM_STEERING_PIN` | "PCA9685.1:40.1" | Direksiyonun bağlı olduğu kanal |
| `PWM_THROTTLE_PIN` | "PCA9685.1:40.0" | Gazın bağlı olduğu kanal |

## PWM kalibrasyon — Arabaya özel sayılar
Her servo/motor farklı çalışır. Bu sayıları `donkey calibrate` ile **ölçersin.**

| Ayar | Ne |
|---|---|
| `STEERING_LEFT_PWM` | Tekerleği tam sola çeviren değer |
| `STEERING_RIGHT_PWM` | Tam sağa çeviren değer |
| `THROTTLE_FORWARD_PWM` | İleri giden değer |
| `THROTTLE_STOPPED_PWM` | Durduğu (nötr) değer |
| `THROTTLE_REVERSE_PWM` | Geri giden değer |

**Neden ölçmek gerek:** Bu sayılar her arabada farklıdır; hazır değer yanlış
direksiyon/hıza yol açar.

## WEB CONTROL + JOYSTICK
Web arayüzü ve elle sürüş ayarları.

| Ayar | Değer | Ne yapar |
|---|---|---|
| `WEB_CONTROL_PORT` | 8887 | Web arayüzü portu → `http://<jetson-ip>:8887` |
| `USE_JOYSTICK_AS_DEFAULT` | True | `--js` yazmadan joystick aktif olur |
| `CONTROLLER_TYPE` | 'xbox' | Joystick türü (ps3\|ps4\|xbox\|F710...) — seninkini yaz |
| `JOYSTICK_MAX_THROTTLE` | 0.5 | Joystick'in verebileceği maks gaz (güvenlik freni) |
| `AUTO_RECORD_ON_THROTTLE` | True | **Gaz verince otomatik kayıt** — veri toplamanın temeli |

## RECORD — Kayıt
| Ayar | Değer | Ne yapar |
|---|---|---|
| `RECORD_DURING_AI` | False | Model sürerken kayıt yapma (manuel toplarken zaten False kalır) |
| `AUTO_CREATE_NEW_TUB` | False | Her sürüşte yeni klasör mü açsın |

## DONKEY_GYM — Simülatör mü gerçek mi?
| Ayar | Değer | Ne yapar |
|---|---|---|
| `DONKEY_GYM` | **False** | Gerçek araç = False. (Simülasyon için True olur, o da simulationconfig.py'de) |
| `DONKEY_SIM_PATH` | .env'den | Simülatör programının yolu (gerçek araçta gerekmez) |

## AI sürüş
| Ayar | Değer | Ne yapar |
|---|---|---|
| `AI_THROTTLE_MULT` | 1.0 | Modelin verdiği gazı topluca ölçekle. Araba hızlıysa <1 yap (örn. 0.7) |

## ⭐ TARGET_POINT bloğu — BİZİM eklediğimiz (en kritik)
Dosyanın sonunda. Modelin "şuraya git" çıktısını **gerçek direksiyon/gaza**
çeviren ayarlar. **Bunlar modelde değil, sürüş anında okunur** → değiştirmek için
yeniden eğitim gerekmez.

| Ayar | Değer | Ne yapar / ne zaman değiştir |
|---|---|---|
| `TARGET_POINT_IMAGE_W/H` | **128** | Model girdi boyutu. **ASLA değiştirme** (model 128 ile eğitildi) |
| `TARGET_POINT_MODEL_ARCH` | 'efficient' | Model mimarisi (dokunma) |
| `TARGET_POINT_INVERT_COLORS` | False | Renk ters çevirme. inverted modelde bile **False** kalır |
| `TARGET_POINT_STEER_GAIN` | 1.35 | Direksiyon şiddeti. **Virajı kaçırırsa artır**, zikzak yaparsa azalt |
| `TARGET_POINT_STEER_SIGN` | 1.0 | Direksiyon **ters kırıyorsa -1.0** yap |
| `TARGET_POINT_TARGET_X_BIAS_M` | 0.08 | Araba **sürekli bir yana çekerse** bununla ortala |
| `TARGET_POINT_BASE_THROTTLE` | 0.18 | Düz yol hızı. **İlk testte düşük tut** |
| `TARGET_POINT_MIN_THROTTLE` | 0.07 | Keskin virajdaki (en yavaş) hız |
| `TARGET_POINT_DYNAMIC_THROTTLE` | True | Virajda otomatik yavaşlama açık |
| `TARGET_POINT_LOOKAHEAD_METERS` | 1.0 | Ne kadar ileri "bakacağı". Geç dönüyorsa azalt |

> Bu blok = arabanın "sürüş karakteri". Araba garip sürüyorsa **önce burayı ayarla**
> (yeniden eğitim yapmadan).

---

# 2) KULLANMADIĞIMIZ AYARLAR (neden var?)

DonkeyCar onlarca farklı donanımı desteklediği için config'de hepsinin ayarı var.
**Sen bunları kullanmıyorsun, dokunmana gerek yok.** Sadece "bu satırlar ne?"
diye merak edersen:

| Bölüm | Ne için (biz kullanmıyoruz ❌) |
|---|---|
| `SERVO_HBRIDGE_*`, `DC_*` | Farklı motor sürücüleri (H-bridge, DC motor) |
| `VESC_*` | VESC motor kontrolcüsü |
| `PIGPIO_PWM`, `STEERING_PWM_PIN` | Eski/alternatif PWM yöntemi |
| `HAVE_ODOM`, `ENCODER_TYPE` | Tekerlek hız sensörü (odometre) |
| `USE_LIDAR`, `LIDAR_*` | LIDAR mesafe sensörü |
| `HAVE_TFMINI` | TFMini mesafe sensörü |
| `HAVE_IMU`, `IMU_*` | İvme/jiroskop sensörü |
| `HAVE_SOMBRERO` | Sombrero HAT kartı |
| `STEERING_RC_GPIO`, `PIGPIO_*` | RC alıcı ile kontrol |
| `MM1_*` | RoboHAT MM1 kartı |
| `USE_SSD1306...` | OLED ekran |
| `HAVE_RGB_LED`, `LED_*` | Durum LED'i |
| `HAVE_MQTT_TELEMETRY`, `TELEMETRY_*` | MQTT telemetri |
| `TRAIN_BEHAVIORS`, `BEHAVIOR_*` | Davranışsal model (şerit seçimi) |
| `TRAIN_LOCALIZER` | Konum tahmini (deneysel) |
| `PATH_FILENAME`, `PATH_SCALE` | GPS yol takibi |
| `AUGMENTATIONS`, `ROI_CROP_*`, `CANNY_*` | DonkeyCar'ın kendi görüntü işleme |
| `BATCH_SIZE`, `MAX_EPOCHS`, `LEARNING_RATE` | DonkeyCar'ın kendi eğitimi (biz `ai_pipeline` kullanıyoruz) |

> Yani bu bölümdeki yüzlerce satır, **senin donanımında yok olan özellikler için.**
> Görmezden gel.

---

# Özet: Gerçek araçta DOKUNULACAK ayarlar

Tüm dosyada sadece şunları ayarlaman yeterli (gerisi varsayılan kalır):

```python
# 1) Kamera (USB)
CAMERA_TYPE = "WEBCAM"
CAMERA_INDEX = 0              # ls /dev/video* ile öğren

# 2) Motor kartı (genelde zaten doğru)
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"
PCA9685_I2C_BUSNUM = 1

# 3) Kalibrasyon (donkey calibrate ile ÖLÇ)
STEERING_LEFT_PWM = ...
STEERING_RIGHT_PWM = ...
THROTTLE_FORWARD_PWM = ...
THROTTLE_STOPPED_PWM = ...

# 4) Joystick
CONTROLLER_TYPE = 'xbox'     # seninkini yaz

# 5) Model (zaten doğru, kontrol et)
TARGET_POINT_IMAGE_W = 128
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_BASE_THROTTLE = 0.18   # ilk testte düşük
```

Adım adım kurulum için → [JETSON_KURULUM.md](02_JETSON_KURULUM.md)
