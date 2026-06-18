# donkeyctl.py — Simülasyon + Tünel + QR Launcher

`donkeyctl.py`, DonkeyCar simülasyonunu tek komutla ayağa kaldıran bir yardımcı
araçtır. Arka planda `manage.py drive` sürecini başlatır, seçtiğin pisti ve modeli
yükler, ardından web arayüzünü (port `8887`) bir tünel üzerinden dışarı açar ve
mobil uygulamanın bağlanması için terminale bir **QR kod** basar.

Tek cümleyle: **"Pisti seç → bağlantı modunu seç → modeli seç → telefonla QR'ı tara → sür."**

---

## Ne işe yarar?

| Görev | Açıklama |
|---|---|
| **Sim'i başlatma** | `manage.py drive` sürecini doğru Python yorumlayıcısı, pist ve model ile çalıştırır. |
| **Tünel açma** | Web arayüzünü `zrok`, `ngrok` veya LAN (aynı WiFi) üzerinden dışarı açar. |
| **QR üretme** | Bağlantı bilgilerini (URL, WebSocket, video, snapshot) JSON olarak QR koda gömer ve terminale basar. |
| **Oturum yönetimi** | Çalışan PID'leri, seçilen pist/model/tüneli `.donkeyctl/session.json` içine kaydeder. |
| **Temiz kapatma** | `stop` ile manage.py, tünel ve simülatör süreçlerini güvenli şekilde sonlandırır. |
| **Durum/QR tekrar** | `status` ve `qr` ile aktif bağlantı bilgisini yeniden gösterir. |

> Bu script, projenin **kullanım (deployment / demo)** tarafıdır; veri toplama veya
> eğitim yapmaz. Eğitim için `ai_pipeline/` araçlarına bak.

---

## Önkoşullar

- **Python ortamı**: Proje kökünde `.venv/bin/python`, ya da `DONKEY_PYTHON` ortam
  değişkeni, ya da aktif `conda/venv`. Script, içinde `donkeycar` ve `docopt` paketleri
  kurulu olan yorumlayıcıyı otomatik bulur.
- **`manage.py`, `simulationconfig.py`, `models/`**: `donkeyctl.py` ile aynı dizinde olmalı.
- **DonkeySim binary**: `simulationconfig.py` içindeki `DONKEY_SIM_PATH` ayarlı olmalı.
- **Tünel araçları** (kullanacağın moda göre):
  - `zrok2` → `~/bin/zrok2` veya `PATH`'te; ilk kullanımdan önce `zrok2 enable`
  - `ngrok` → `brew install ngrok` + authtoken
  - LAN → ek araç gerekmez, sadece aynı WiFi
- **QR için** (opsiyonel): `pip install qrcode` — yoksa script payload JSON'ı düz metin basar.

---

## Komutlar

```bash
python donkeyctl.py start                  # pist + bağlantı modu + model menüsü (interaktif)
python donkeyctl.py start --model dilara   # belirli bir modelle başlat
python donkeyctl.py start --tunnel ngrok   # menüsüz, ngrok ile
python donkeyctl.py start --tunnel lan     # aynı WiFi, IP ile QR
python donkeyctl.py start --tunnel lan --lan-ip 192.168.1.42
python donkeyctl.py start --no-model       # AI olmadan manuel sürüş
python donkeyctl.py start -y               # tüm menüleri atla (session/varsayılan kullan)

python donkeyctl.py restart                # aynı ayarlarla hard refresh
python donkeyctl.py stop                   # tüm servisleri durdur
python donkeyctl.py status                 # aktif bağlantı bilgilerini göster
python donkeyctl.py qr                     # QR kodunu terminale yeniden bas
python donkeyctl.py tracks                 # desteklenen pist listesi
python donkeyctl.py models                 # models/ altındaki modeller
```

### `start` parametreleri

| Parametre | Açıklama |
|---|---|
| `--track <id>` | Pist ID'si (ör. `donkey-warren-track-v0`). Verilmezse menüden seçilir. |
| `--tunnel <mod>` | `zrok` \| `ngrok` \| `lan` \| `none`. Verilmezse menüden seçilir. |
| `--lan-ip <ip>` | LAN modunda kullanılacak IPv4 (otomatik algılamayı geçersiz kılar). |
| `--no-ngrok` | `--tunnel none` ile aynı (sadece localhost). |
| `--model <ad/yol>` | Model adı veya yolu (ör. `my_first_pilot` veya `models/model.keras`). |
| `--no-model` | Modelsiz başlat — manuel sürüş. |
| `--record` | Gaz verildikçe otomatik kaydı açar (`AUTO_RECORD_ON_THROTTLE=true`). |
| `--port <n>` | Web kontrol portu (varsayılan `8887`). |
| `-y`, `--yes` | Pist/tünel/model menülerini atla; session veya varsayılanları kullan. |

---

## Bağlantı modları (`--tunnel`)

| Mod | Ne zaman | Notlar |
|---|---|---|
| **`zrok`** | Uzaktan erişim (varsayılan) | `~/bin/zrok2` + `zrok2 enable` gerekir. Free: 5 GB/gün. |
| **`ngrok`** | Uzaktan erişim alternatifi | authtoken gerekir. Free: 1 GB/ay çıkış, ~20K HTTP/ay, 3 online endpoint. |
| **`lan`** | Aynı WiFi'deki telefon | Tünel yok, kota yok. IP otomatik algılanır; güvenlik duvarı engelleyebilir. |
| **`none`** | Sadece bu bilgisayar | Public URL yok, yalnızca `localhost:8887`. |

İnteraktif menüde her modun yanında **kurulu mu / kota durumu** gibi ipuçları gösterilir.
Varsayılan mod `DONKEY_TUNNEL` ortam değişkeni ile değiştirilebilir.

---

## Ne üretir? (QR payload)

Başarılı bir `start` sonrası mobil uygulamaya gönderilen JSON şuna benzer:

```json
{
  "v": 1,
  "base": "https://abc123.share.zrok.io",
  "track": "donkey-warren-track-v0",
  "tunnel": "zrok",
  "ws": "wss://abc123.share.zrok.io/wsDrive",
  "video": "https://abc123.share.zrok.io/video",
  "snapshot": "https://abc123.share.zrok.io/snapshot",
  "drive": "https://abc123.share.zrok.io/drive",
  "port": 8887,
  "model": "models/model.keras"
}
```

- `ws` → telefonun gerçek zamanlı kontrol için bağlandığı WebSocket
- `snapshot` → React Native tarafında önerilen kamera akışı uç noktası
- `video` → MJPEG video akışı

---

## Çalışma akışı (start → telefonla bağlan)

```
python donkeyctl.py start
        │
        ▼
1. Doğru Python'u bul (.venv / DONKEY_PYTHON / conda)
2. Pist seç  →  bağlantı modu seç  →  model seç
3. Eski süreçleri temizle (manage.py / tünel / sim)
4. manage.py drive başlat → localhost:8887/drive ayağa kalkana kadar bekle
5. Tüneli aç (zrok/ngrok/lan) → public URL hazır olana kadar bekle
6. Bağlantı bilgilerini yazdır + QR kodu terminale bas
7. session.json'a kaydet
        │
        ▼
Telefonla QR'ı tara → sür
```

Herhangi bir adım başarısız olursa script **rollback** yapar: açtığı tüm süreçleri
kapatır ki yarım kalmış bir tünel/sim arkada çalışmasın.

---

## Oturum durumu (`.donkeyctl/`)

| Yol | İçerik |
|---|---|
| `.donkeyctl/session.json` | Son çalışmanın PID'leri, pist, model, tünel, bağlantı payload'u. |
| `.donkeyctl/donkey_qr.png` | QR kodunun yedek PNG dosyası. |
| `.donkeyctl/logs/manage.log` | `manage.py drive` çıktısı. |
| `.donkeyctl/logs/zrok.log` / `ngrok.log` | Tünel süreç logları. |

Sorun yaşarsan **ilk bakılacak yer bu loglardır.**

---

## Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `Donkey Python bulunamadı` | `conda activate donkey` ya da `DONKEY_PYTHON=/yol/python` ayarla. |
| `zrok2 bulunamadi` | `~/bin/zrok2` kur veya `PATH`'e ekle. |
| `zrok2 ortami aktif degil` | Bir kere `zrok2 enable` çalıştır. |
| `ngrok bulunamadı` | `brew install ngrok` + authtoken ekle. |
| `manage.py 8887 portunda ayağa kalkmadı` | `.donkeyctl/logs/manage.log` dosyasına bak (sim yolu / model hatası). |
| LAN URL yanıt vermiyor | macOS güvenlik duvarı Python'u engelliyor olabilir; ya da `--lan-ip` ile doğru IP'yi ver; telefon aynı WiFi'de olmalı. |
| Public URL 502 / bad gateway | Script otomatik rollback yapar; `python donkeyctl.py start` ile tekrar dene. |

---

## İlgili dokümanlar

- [00_PIPELINE_REHBER.md](00_PIPELINE_REHBER.md) — projenin genel akışı
- [04_CONFIG_REHBERI.md](04_CONFIG_REHBERI.md) — `simulationconfig.py` / `config.py` ayarları
- Proje kökü: `CALISTIRMA_REHBERI.md` — uçtan uca çalıştırma rehberi
