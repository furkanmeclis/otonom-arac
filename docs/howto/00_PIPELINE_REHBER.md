# AI Pipeline Rehberi — Sıfırdan Anlatım

Bu rehber projeyi **hiç bilmeyen birine** anlatır gibi yazıldı. Her aşamada
"ne oluyor, neden oluyor, ne girip ne çıkıyor ve hangi dosya bunu yapıyor"
sorularını yanıtlar. Kod fiziksel olarak taşınmadı; bu sadece bir "harita".

---

## Önce: Bu proje aslında ne yapıyor?

Bir oyuncak araba var. Üstünde bir kamera. Amaç: araba kameradan gördüğüyle
**kendi kendine pisti sürebilsin.**

Çoğu projede model şunu öğrenir: "şu görüntüyü gördüm → direksiyonu şu kadar
çevir". Bu projede ise **farklı bir yol** seçilmiş. Model direksiyonu değil,
**"nereye gitmeliyim?"** sorusunu yanıtlıyor. Yani önündeki yolda gitmesi
gereken bir **hedef nokta** söylüyor:

```
        (hedef nokta)
            o
           /
          /   ← araba "şuraya gitmeliyim" diyor
        [araba]
```

Bu hedef nokta `(target_x, target_y)` diye iki sayıdır:
- `target_y` = kaç metre **ileri** (örn. 1.2 m)
- `target_x` = kaç metre **sağa/sola** (örn. +0.3 m sağ, -0.2 m sol)

Sonra ayrı bir "kontrolcü" (sürücü mantığı) şunu hesaplar: "Bu noktaya gitmek
için direksiyonu ne kadar çevirmeliyim?" Bunu basit geometriyle yapar.

**Neden böyle?** Çünkü "nereye gitmeliyim" sorusu, arabanın fiziğinden (lastik,
motor, ağırlık) bağımsızdır. Bir hedef noktayı simülasyondaki araba da, gerçek
araba da aynı şekilde anlar. Bu yüzden simülasyonda öğrenileni gerçeğe aktarmak
(sim-to-real) daha kolay olur.

---

## Tüm akış tek bakışta

```
0. KURULUM
       │
       ▼
1. VERİ TOPLAMA        Simülatörde arabayı sürüp binlerce kare topla.
   (collect_...)       Her kare = 1 fotoğraf + o an "doğru hedef nokta" neresiydi.
       │
       ▼
2. ETİKETLEME          Ham kareleri eğitime hazır tabloya (manifest) çevir.
   (build_...labels)   "Bu fotoğrafın doğru cevabı (target_x, target_y) = (...)"
       │
       ▼
3. EĞİTİM              Yapay sinir ağına bu örnekleri göster, öğrensin.
   (train.py)          Çıktı: model.keras (eğitilmiş beyin)
       │
       ▼
4. DEĞERLENDİRME       Modeli simülatöre koy, gerçekten sürebiliyor mu bak.
   (evaluate_...)      "10 turdan kaçını tamamladı? Kaç kez yoldan çıktı?"
       │
       ▼
5. SÜRÜŞ               Modeli gerçek arabaya/simülatöre koy ve sür.
   (manage.py drive)
```

Şimdi her aşamayı tek tek, detaylı açalım.

---

## 0) KURULUM

Bilgisayarı hazırlama. Python'un **tam olarak 3.11** sürümü gerekiyor (TensorFlow
2.15.1 bunu istiyor; başka sürümle çalışmaz).

```powershell
py -3.11 -m venv .venv          # izole bir Python ortamı oluştur
.venv\Scripts\activate          # o ortamı aç
pip install -r requirements-train.txt   # gerekli kütüphaneleri kur
```

Sonra `simulationconfig.py` dosyasında `DONKEY_SIM_PATH`'i, simülatör
programının yerini gösterecek şekilde ayarla. Her şey yerli yerinde mi diye
hızlı test:

```powershell
python manage.py smoke --simulationconfig=simulationconfig.py
```

---

## 1) VERİ TOPLAMA

### Bu aşamada ne oluyor?

Yapay zekâ örnekten öğrenir. Bir çocuğa araba sürmeyi öğretmek gibi: önce
**doğru sürüşü göstermen** lazım. Burada da simülatörde arabayı sürüp binlerce
"durum → doğru hedef nokta" örneği topluyoruz.

Ama burada arabayı **sen sürmüyorsun.** Onun yerine "teacher" (öğretmen) denen
bir otomatik sürücü sürüyor. Teacher'ın bir avantajı var: simülatör ona arabanın
**gerçek konumunu ve pistin tam haritasını** veriyor (gerçek hayatta bu
bilgiler yok). Teacher bu ayrıcalıklı bilgiyle her an "doğru hedef nokta
neresiydi" hesaplayabiliyor. Model ise sadece fotoğrafı görecek — yani
teacher'ın bildiğini fotoğraftan tahmin etmeyi öğrenecek.

Veri toplama **3 aşamada** yapılır ve sırası önemli:

**Aşama 1 — Harita çıkarma (`map`):**
Teacher pisti bir kez sakin, ortadan, temiz sürer. Bu sürüş sırasında arabanın
geçtiği her noktayı kaydeder. Bu noktalar birleşince pistin **orta çizgisi
(centerline)** ortaya çıkar. Bu, daha sonra "doğru cevapları" hesaplayacağımız
referans yoldur. Harita olmadan etiket üretilemez.

➤ **Bu adımı yapan komut:**
```powershell
python ai_pipeline/collect_target_point_data.py --task map --track generated_roads --simulationconfig=simulationconfig.py
```

**Aşama 2 — Düşük gürültülü sürüş (`phase2_low_noise`):**
Teacher pisti tekrar sürer ama bu sefer **fotoğrafları da kaydeder** ve hafif
rastgelelik ekler. Bu, temiz ve bol miktarda "normal sürüş" verisi üretir.

➤ **Bu adımı yapan komut:**
```powershell
python ai_pipeline/collect_target_point_data.py --task collect --collection-profile phase2_low_noise --track generated_roads --simulationconfig=simulationconfig.py
```

**Aşama 3 — Tam gürültü / kurtarma (`phase3_full_noise`):**
Teacher bilerek arabayı kenara kaydırır, sonra merkeze geri döner. Buna
**kurtarma (recovery)** denir. Çok önemli: eğer model sadece mükemmel sürüşü
görürse, bir gün küçük bir hata yapıp kenara kayınca **ne yapacağını bilemez**
ve yoldan çıkar. Kurtarma verisi ona "hata yaptıysan böyle düzel" diye öğretir.

➤ **Bu adımı yapan komut:**
```powershell
python ai_pipeline/collect_target_point_data.py --task collect --collection-profile phase3_full_noise --track generated_roads --simulationconfig=simulationconfig.py
```

> Not: Bu üç komut **sırayla** çalıştırılır — önce harita, sonra phase2, sonra
> phase3. `--track` ile pist adını değiştirebilirsin (örn. `mini_monaco`).

### Girdi → Çıktı

```
Girdi : Simülatör (pist)
Çıktı : data/sim/.../  klasörü dolusu:
          - images/        → 1.jpg, 2.jpg, 3.jpg ...  (her kare bir fotoğraf)
          - raw kayıtlar   → her fotoğraf için: konum, yön, hız, teacher'ın
                             ürettiği doğru hedef nokta, uygulanan komut
```

### Hangi dosya ne yapıyor?

- **`collect_target_point_data.py`** — Senin çalıştırdığın **ana komut.** Hangi
  aşamayı (map/collect) ve hangi profili istediğini söylersin, o da gerekli
  parçaları çağırıp veriyi toplar.

- **`target_point/mapping.py`** — Aşama 1'in beyni. Teacher'ı pistte sürdürür,
  geçtiği noktaları toplar ve orta çizgiyi (haritayı) diske yazar.

- **`target_point/teacher_policy.py`** — İki şey içerir: (a) simülatörü **süren**
  otomatik sürücüler, (b) her an "doğru hedef nokta neresi" diye hesaplayan
  **etiketleme mantığı.** Projenin en kritik dosyalarından biri.

- **`target_point/collector.py`** — Aşama 2/3'ün beyni. Teacher ile sürüp her
  kareyi (fotoğraf + doğru etiket) diske kaydeder. Ayrıca "eğitilmiş modeli
  sürdürüp, hata yaptığı yerlerde teacher'ın doğrusunu kaydetme" (rollout/DAgger)
  yöntemi de buradadır.

- **`target_point/track_map.py`** — Harita veri yapıları. Haritayı diske
  `metadata.json + raw_trace.csv + centerline.csv` olarak yazar ve geri okur.

- **`target_point/sim_session.py`** — Simülatörle konuşan alt katman. "Şu komutu
  uygula" der, karşılığında "yeni fotoğraf + arabanın konumu" alır.

---

## 2) ETİKETLEME

### Bu aşamada ne oluyor?

1. aşamada topladığımız veri "ham" haldedir — dağınık fotoğraflar ve kayıtlar.
Eğitimin bunu doğrudan okuması zordur. Bu aşamada hepsini düzenli, tek tip bir
**tabloya** (manifest dosyası) çeviriyoruz. Tablonun her satırı bir eğitim
örneğidir:

```
fotoğrafın yolu          | target_x | target_y | senaryo  | ...
images/1532.jpg          |   0.31   |   1.20   | turn     | ...
images/1533.jpg          |  -0.05   |   1.20   | straight | ...
```

Bu tabloya **manifest** (JSONL formatında) denir. "Şu fotoğrafın doğru cevabı
şudur" listesidir.

Bu aşamada ayrıca **temizlik ve dengeleme** yapılır:
- Bozuk kareler (eksik bilgi, saçma değerler) atılır.
- **Dengeleme:** Sürüşün çoğu "düz git"tir. Eğer veriyi olduğu gibi bırakırsak
  model sadece düz gitmeyi öğrenir, virajları/kurtarmayı ihmal eder. Bu yüzden
  düz örneklerin bir kısmı azaltılır, viraj/kurtarma örneklerinin payı korunur.

### Girdi → Çıktı

```
Girdi : data/sim/.../  (1. aşamanın ham verisi)
Çıktı : data/sim_multitrack/index/  içinde JSONL manifest dosyaları
        (train ve val olarak ayrılmış, etiket modlarına göre)
```

### Hangi dosya ne yapıyor?

- **`build_target_point_labels.py`** — Senin çalıştırdığın **ana komut.** Ham
  veriyi manifest'e dönüştürür.

- **`target_point/manifest.py`** — İşin asıl yapıldığı yer: kareleri indeksler,
  bozukları filtreler, senaryoları **dengeler** ve manifest'i yazar.

- **`target_point/external_adapter.py`** — **GERÇEK ARABA** verisi için özel.
  Gerçekte pist haritası/teacher olmadığından "doğru hedef nokta" bilinmez.
  Bu dosya bir hile yapar: sen sürerken kaydedilen **direksiyon açısından geriye
  doğru** bir hedef nokta uydurur (pseudo-label). Bu, sim verisi kadar iyi
  değildir ama gerçek görüntülerle çalışmanın tek yoludur.

### Komut

```powershell
python ai_pipeline/build_target_point_labels.py --raw-roots data/sim/generated_roads --output-dir data/sim_multitrack/index
```

> **Önemli — Sim vs Gerçek araba farkı:**
> - **Simülasyonda:** etiket, harita + teacher'dan gelir → bu adım gereklidir ve
>   etiketler "gerçek geometri"dir (kaliteli).
> - **Gerçek arabada:** harita yoktur. `external_adapter.py` direksiyondan
>   pseudo-etiket üretir ve bu, eğitim sırasında otomatik olur (ayrı komut yok).

---

## 3) EĞİTİM

### Bu aşamada ne oluyor?

Asıl öğrenmenin olduğu yer. Manifest'teki binlerce "fotoğraf → doğru hedef
nokta" örneğini **yapay sinir ağına (CNN)** tekrar tekrar gösteriyoruz. Ağ önce
rastgele tahminler yapar, çok yanılır; her yanılgıda kendini biraz düzeltir.
Yeterince tekrar sonra fotoğraftan hedef noktayı doğru tahmin etmeye başlar.

Birkaç önemli kavram:

- **CNN (evrişimli sinir ağı):** Görüntüden desen çıkaran ağ türü. Bu projede
  küçük tutulmuş (~33 bin parametre) çünkü gerçek arabadaki küçük bilgisayarda
  (Jetson Nano) hızlı çalışması gerekiyor.

- **Loss (kayıp):** "Model ne kadar yanılıyor" ölçüsü. Eğitim bunu küçültmeye
  çalışır. Burada `target_x` hatasına `target_y`'den daha çok ceza verilir,
  çünkü sağa/sola sapma (x) direksiyonu doğrudan etkiler.

- **Veri artırma (augmentation):** Aynı fotoğrafı hafifçe değiştirip (parlaklık,
  küçük dönme...) çoğaltma. Modelin ezberlemesini önler, gerçek dünyaya
  dayanıklılığını artırır.

- **Çökme (collapse):** Modelin tehlikeli bir tembelliği — fotoğrafa bakmadan
  hep aynı (ortalama) noktayı söylemeye başlaması. Loss düşük görünür ama model
  işe yaramaz. Bir izleyici (monitor) bunu yakalar.

### Girdi → Çıktı

```
Girdi : Manifest (2. aşama) + bir config dosyası (hangi strateji?)
Çıktı : models/model.keras  ← eğitilmiş "beyin"
```

### Config dosyaları nedir? (`configs/` altında ne var?)

Aynı eğitim kodunu **farklı stratejilerle** çalıştırmak için hazır ayar
dosyaları. Her biri "hangi veriyle, ne kadar veri artırmayla, hangi model
boyutuyla eğiteceğiz" gibi onlarca ayarı tek dosyada toplar. Kodu değiştirmeden
sadece config'i değiştirerek farklı denemeler yaparsın.

Bir config dosyası aslında `simulationconfig.py`'yi temel alır, sonra üzerine
kendi ayarlarını yazar (örneğin "veriyi şuradan al", "veri artırmayı kapat").

`configs/` altındaki 9 dosya:

| Config | Stratejisi |
|---|---|
| `model_01_pure_sim.py` | Sadece simülasyon verisi, veri artırma yok. En temel başlangıç. |
| `model_02_sim_domain_randomization.py` | Sim verisi + agresif görsel çeşitlilik (sim-to-real için). |
| `model_03_pure_real.py` | Sadece gerçek araba verisi (pseudo-etiketli). |
| `model_04_hybrid_v1_naive_mix.py` | %70 sim / %30 gerçek (başarısız işaretli — naif karışım çalışmadı). |
| `model_05_hybrid_v2_sim_heavy.py` | %90 sim / %10 gerçek. |
| `model_06_hybrid_v3_real_heavy.py` | %30 sim / %70 gerçek. |
| `model_07_finetune.py` | Önce simde öğren, sonra gerçek veriyle ince ayar. Sim-to-real için en iyisi. |
| `model_11_multitask.py` | Aynı anda direksiyon + gaz tahmini (çok görevli). |
| `model_12_temporal.py` | 5 kareyi birlikte gören (LSTM) zamansal model. |

> **Önemli:** Bazı config'ler (özellikle gerçek/hibrit olanlar) **veriyi nereden
> alacaklarını kendi içlerinde tanımlar.** Örneğin `model_03_pure_real.py` dosyası
> içinde "gerçek tub'lar şu klasörde" yazar. Yani veri seçimi kısmen komuttaki
> `--manifest`, kısmen de config dosyasının içindeki ayarlarla olur (aşağıda).

### Hangi dosya ne yapıyor?

- **`train.py`** — Senin çalıştırdığın **ana komut.** GPU'yu hazırlar ve eğitimi
  başlatır.

- **`target_point/training.py`** — Eğitim döngüsünün kendisi: veriyi yükler,
  etiketleri normalize eder, modeli kurar, eğitir, en iyi sonucu kaydeder.

- **`target_point/dataset.py`** — Manifest'i (veya ham tub'ı) okuyup eğitim
  örnekleri üretir. Sim ve gerçek veriyi karıştırma, train/val'a **sızıntısız
  bölme** (aynı sahnenin kareleri hem train hem val'e düşmesin) burada yapılır.

- **`target_point/model.py`** — Sinir ağının mimarisi + `preprocess_image`
  (görüntüyü kırp/boyutlandır/normalize). Bu ön-işleme eğitimde ve gerçek
  sürüşte **birebir aynı** olmalı, yoksa model gerçekte başka şey görür.

- **`target_point/augment.py`** — Veri artırma (sadece eğitim sırasında).

- **`target_point/domain_randomization.py`** — Sim-to-real için her sürüşe farklı
  ışık/renk/doku "teması" uygulayıp çeşitlilik katar.

- **`target_point/diagnostics.py`** — Eğitim sonrası sağlık kontrolü: hata,
  korelasyon ve **çökme kontrolü** (özellikle virajları öğrendi mi?).

- **`target_point/experiments.py`** — Her eğitim koşusunun sonuçlarını düzenli
  klasörlere/JSON'lara yazar ki farklı config'leri karşılaştırabilesin.

### Komut ve her parçasının anlamı

```powershell
python ai_pipeline/train.py --type target_point --manifest data/sim_multitrack/index --model models/model.keras --label-mode adaptive_v1 --simulationconfig=configs/model_01_pure_sim.py
```

Bu uzun komutu parçalara ayıralım:

| Parça | Ne işe yarar? |
|---|---|
| `python ai_pipeline/train.py` | Eğitim programını çalıştırır. |
| `--type target_point` | **Model türü.** Bu projenin yöntemi olan "hedef nokta" modeli. (Diğer türler: `linear`, `rnn` — başka deney türleri için.) |
| `--manifest data/sim_multitrack/index` | **EĞİTİM VERİSİ — evet, burası senin verini seçtiğin yer.** 2. aşamada (etiketleme) üretilen **manifest klasörünü** gösterirsin. Dikkat: ham fotoğraf klasörünü (`data/sim/...`) değil, etiketlenmiş **`.../index`** klasörünü verirsin. |
| `--model models/model.keras` | **ÇIKTI.** Eğitilen modelin nereye, hangi isimle kaydedileceği. İstediğin ismi verebilirsin (örn. `models/deneme1.keras`). |
| `--label-mode adaptive_v1` | **Etiket modu.** `adaptive_v1` = hedef noktanın ne kadar ileride olacağı hıza/viraja göre değişir. `fixed_1p2m` = hep 1.2 m ileri. (Veriyi 2. aşamada iki modda da ürettiysen burada hangisini kullanacağını seçersin.) |
| `--simulationconfig=configs/model_01_pure_sim.py` | **STRATEJİ (config).** Yukarıdaki tablodan hangi deneme reçetesini kullanacağın. Hangi veri karışımı, ne kadar veri artırma, hangi model boyutu hep buradan gelir. |


---

## 4) DEĞERLENDİRME

### Bu aşamada ne oluyor?

Eğitim bittiğinde modelin "kâğıt üzerindeki" başarısı (loss) iyi görünebilir ama
bu **gerçekten sürebildiği anlamına gelmez.** Tek bir karede iyi tahmin yapmak
başka, yüzlerce kare boyunca arabayı pistte tutmak başkadır (küçük hatalar
birikip büyür).

Bu yüzden modeli simülatöre koyup **gerçekten sürdürürüz** (buna "kapalı-döngü"
denir: model sürer → araba hareket eder → yeni görüntü gelir → model yine sürer).
Sonra ölçeriz:
- Kaç turu tamamladı?
- Kaç kez yoldan çıktı / çarptı?
- Ne kadar yol gitti? Direksiyonu titretiyor mu?

Asıl "iyi model mi" kararını burası verir.

### Girdi → Çıktı

```
Girdi : models/model.keras + simülatör
Çıktı : Başarı raporu (tur tamamlama oranı, pist-dışı sayısı, vb.)
```

### Hangi dosya ne yapıyor?

- **`evaluate_target_point.py`** — Senin çalıştırdığın **ana komut.**
- **`target_point/evaluate_closed_loop.py`** — İşin çekirdeği: modeli pilot +
  kontrolcü ile sürdürür ve metrikleri toplar.

### Komut

```powershell
python ai_pipeline/evaluate_target_point.py --model models/model.keras --tracks generated_roads,mini_monaco --episodes-per-track 10 --simulationconfig=simulationconfig.py
```

---

## 5) SÜRÜŞ (Gerçek Kullanım)

### Bu aşamada ne oluyor?

Artık eğitilmiş ve test edilmiş model var. Onu gerçek arabaya (Jetson Nano +
kamera) veya simülatöre koyup sürdürüyoruz. Akış şöyle döner, saniyede onlarca kez:

```
kamera → fotoğraf → [model: hedef noktayı tahmin et] → (target_x, target_y)
       → [kontrolcü: bunu direksiyon/gaza çevir] → araba hareket eder → tekrar
```

### Hangi dosya ne yapıyor?

- **`../manage.py`** — Senin çalıştırdığın **ana komut** (kök dizinde). Kamera,
  model, joystick gibi tüm parçaları birbirine bağlar ve arabayı sürer.

- **`target_point/pilot.py`** — "Pilot": kameradan gelen fotoğrafı modele verip
  hedef noktayı alan parça. Keras (.keras) modelini kullanır.

- **`target_point/pilot_tflite.py`** — Aynı pilotun Jetson için hızlı sürümü.
  Model "nicelenmiş" (.tflite) halde olduğundan küçük bilgisayarda hızlı koşar.

- **`target_point/controller.py`** — Hedef noktayı **direksiyon ve gaza** çeviren
  geometrik kontrolcü. Ayrıca virajda yavaşlama, direksiyon yumuşatma gibi
  akıllı davranışları ekler.

- **`target_point/export.py`** — Eğitilmiş .keras modelini Jetson için
  `.tflite` formatına (INT8/FP16) dönüştürür.

### Komut (simülatörde sürüş)

```powershell
python manage.py drive --model models/model.keras --type target_point --simulationconfig=simulationconfig.py
```

---

## 6) GERÇEK HAYAT: JETSON NANO'DA SÜRÜŞ

Buraya kadar her şey bilgisayarında/simülatörde oldu. Şimdi modeli **gerçek
arabaya** (Jetson Nano + kamera + motor) koyup sürdürmek istiyorsun. Burada
simülatörden iki temel fark var:

1. **Config değişir.** Simülatörde `simulationconfig.py` (içinde `DONKEY_GYM = True`)
   kullandın. Gerçek arabada **`config.py`**'yi kullanırsın — orada kamera tipi,
   motor pinleri ve kalibrasyon değerleri vardır, `DONKEY_GYM` kapalıdır.
2. **Donanım gerçektir.** Kamera gerçek ışığı görür, motor gerçek tekerleği
   döndürür. Yani kalibrasyon (direksiyonun PWM değerleri) önem kazanır.

### Adım adım

**Adım 1 — Modeli Jetson'a taşı ve hızlı formata çevir.**
Eğittiğin `model.keras`'ı Jetson'a kopyala. Jetson küçük bir bilgisayar
olduğundan, Keras modelini doğrudan koşturmak yavaş olabilir; bunu önce
**TFLite**'a çevirmen önerilir (daha hızlı, daha küçük):

```powershell
python -c "from target_point.export import export_tflite; export_tflite('models/model.keras', 'models/model.tflite', quantize='float16')"
```

> `quantize='int8'` daha da hızlıdır ama kalibrasyon görüntüsü ister; başlangıçta
> `float16` en kolayı.

**Adım 2 — Donanımı kalibre et (`config.py`).**
Arabanın direksiyonu ve gazı PWM sinyaliyle sürülür. Her arabanın değerleri
farklıdır. DonkeyCar'ın hazır aracıyla ölç ve `config.py`'ye yaz:

```powershell
donkey calibrate --channel 1 --bus 1   # direksiyon: tam sol/sağ PWM değerlerini bul
donkey calibrate --channel 0 --bus 1   # gaz: ileri PWM değerini bul
```

Bulduğun değerleri `config.py`'de güncelle:
```python
STEERING_LEFT_PWM = 460     # senin arabanın tam sol değeri
STEERING_RIGHT_PWM = 290    # tam sağ
THROTTLE_FORWARD_PWM = 500  # ileri
CAMERA_TYPE = "CSIC"        # Jetson CSI kamera için (USB ise "WEBCAM")
```

**Adım 3 — Kameranın eğitimdekine benzemesini sağla.**
Bu KRİTİK: model, eğitimde gördüğü açıyı bekler. Gerçek kamerayı arabaya
**eğitimdeki simülatör kamerasıyla benzer yükseklik ve açıda** monte et. Ön-işleme
ayarları (`TARGET_POINT_IMAGE_W/H`, kırpma) eğitimle aynı kalmalı — bunlara
dokunma.

**Adım 4 — Sür.**
Gerçek arabada `--simulationconfig` vermezsin; varsayılan `config.py` kullanılır:

```powershell
python manage.py drive --model models/model.tflite --type target_point
```

> İlk denemede **gazı çok düşük tut** ve joystick'i elinde hazır bulundur ki
> araba kaçarsa hemen müdahale edebilesin. Web arayüzü `http://<jetson-ip>:8887`
> adresinden de izlenebilir.

### Sürüşü ayarlayan parametreler (config.py, eğitim GEREKTİRMEZ)

Bunlar modelin içinde değil, kontrolcüdedir — yani değiştirmek için yeniden
eğitmen GEREKMEZ, sadece config'i düzenleyip tekrar sürersin:

| Parametre | Ne yapar? | Belirti → çözüm |
|---|---|---|
| `TARGET_POINT_STEER_SIGN` | Direksiyon yönü (+1 / -1) | Araba sola derken sağa kırıyorsa `-1` yap. |
| `TARGET_POINT_STEER_GAIN` | Direksiyon şiddeti (varsayılan 1.35) | Virajları kaçırıyorsa artır; zikzak yapıyorsa azalt. |
| `TARGET_POINT_TARGET_X_BIAS_M` | Sabit yanal kayma düzeltmesi | Hep bir yana çekiyorsa bununla ortala. |
| `TARGET_POINT_THROTTLE` | Temel gaz (varsayılan 0.2) | Çok hızlı/yavaşsa ayarla. |
| `TARGET_POINT_DYNAMIC_THROTTLE` | Virajda otomatik yavaşlama | Açıkken viraja sakin girer. |
| `TARGET_POINT_LOOKAHEAD_METERS` | Ne kadar ileri bakacağı | Geç dönüyorsa azalt, titriyorsa artır. |

---

## 7) MODEL KÖTÜ SÜRERSE NE YAPMALI? (Panik yapma — her şeyi baştan yapma)

Bu senin asıl endişen: "Ya simülasyonda eğittiğim model gerçekte kötü sürerse,
baştan veri toplayıp eğitmem mi gerekecek?"

**Kısa cevap: Büyük ihtimalle HAYIR.** Kötü sürüş ≠ kötü model. Çoğu zaman sorun
kalibrasyon veya ayardır; bunlar dakikalar içinde, **yeniden eğitim olmadan**
çözülür. Önce ucuz/hızlı çözümleri dene, sıfırdan toplama **en son** çaredir.

### Çözüm merdiveni (ucuzdan pahalıya)

```
1. KALİBRASYON          (dakikalar, eğitim YOK)
   - Direksiyon ters mi? → STEER_SIGN = -1
   - Az/çok mu dönüyor?  → STEER_GAIN ayarla
   - Hep bir yana mı?    → TARGET_X_BIAS_M ayarla
   - PWM doğru mu?       → donkey calibrate
        │ hâlâ kötüyse ▼
2. KAMERA / ÖN-İŞLEME    (dakikalar, eğitim YOK)
   - Kamera açısı/yüksekliği eğitimdekine benziyor mu?
   - Görüntü boyutu/kırpma eğitimle aynı mı?
        │ hâlâ kötüyse ▼
3. KONTROLCÜ AYARI       (dakikalar, eğitim YOK)
   - LOOKAHEAD, DYNAMIC_THROTTLE, gaz seviyesini ayarla
   - Hızı düşür (yavaşta her model daha iyi sürer)
        │ hâlâ kötüyse VE sebep görsel fark (ışık/doku/zemin) ise ▼
4. KÜÇÜK GERÇEK VERİ + FİNE-TUNE   (saatler, AZ eğitim — model_07)
   - Arabayı 10-20 dk MANUEL sür, veri topla (sıfırdan DEĞİL)
   - Sim'de eğitilmiş modeli bu az gerçek veriyle "ince ayar" yap
        │ çok büyük fark varsa, gelecekte ▼
5. SİM VERİSİNİ İYİLEŞTİR   (günler — en son çare)
   - Domain randomization (model_02): sim'i daha çeşitli yap
   - Yeni pist/senaryo ekle
```

### 4. adımı biraz açalım (en olası gerçek çözüm)

Senin durumun şu: "Sim'de eğittim, gerçekte yetersiz olabilir." Bu yaygın ve
projenin tam da bunun için bir cevabı var: **fine-tuning (`model_07`).**

- Sıfırdan veri toplamana gerek yok. Gerçek arabayı kendin **kısa süre** (örn.
  10-20 dk) joystick'le sür — bu sırada veri (`tub`) kaydedilir.
- Bu küçük gerçek veriyi, sim'de öğrenmiş modelin üstüne **ince ayar** olarak
  eklersin. Model sim'den gelen "sürüş bilgisini" korur, gerçek kameranın
  görünümüne uyum sağlar.
- Komut (gerçek tub'larını config'te gösterdikten sonra):

```powershell
python ai_pipeline/train.py --type target_point --model models/model_07.keras --label-mode adaptive_v1 --simulationconfig=configs/model_07_finetune.py
```

> Gerçek veriyle eğitimin nasıl çalıştığı (ve neden sim'deki kadar "saf" etiket
> olmadığı) `external_adapter.py` bölümünde anlatıldı: gerçekte hedef nokta
> direksiyondan geriye doğru üretilir (pseudo-etiket).

### Maliyet karşılaştırması (neden endişeye gerek yok)

| Çözüm | Süre | Yeniden eğitim? |
|---|---|---|
| Kalibrasyon / kontrolcü ayarı | Dakikalar | Hayır |
| Küçük gerçek veri + fine-tune | Birkaç saat | Az (ince ayar) |
| Sıfırdan veri toplama + eğitim | Günler | Evet (en son çare) |

Yani: elindeki sim modelleri "boşa gitmez." Kötü sürse bile büyük ihtimalle
1–3. adımlarla düzelir; düzelmezse 4. adım (az gerçek veri + fine-tune) neredeyse
her zaman yeter. 5. adıma (sıfırdan) çok nadiren gerek kalır.

---

## Yardımcı dosyalar

- **`target_point/__init__.py`** — Paketin giriş kapısı. En çok kullanılan
  sınıf/fonksiyonlara kısa yoldan erişim sağlar; ağır kütüphaneleri (TensorFlow)
  ancak gerçekten gerekince yükler.
- **`tools/`** — Çeşitli yardımcı betikler.

---

## Sık sorulan: Neden dosyalar `veritoplama/`, `egitim/`, `surus/` klasörlerine bölünmedi?

Mantıklı bir istek ama teknik olarak sorunlu, çünkü `target_point/` içindeki
modüllerin çoğu **tek bir aşamaya ait değil.** Örnek:

| Modül | Veri toplama | Eğitim | Sürüş |
|---|:---:|:---:|:---:|
| `model.py` | ✓ (rollout) | ✓ | ✓ (pilot) |
| `controller.py` | | | ✓ (+ değerlendirme) |
| `preprocess_image` (model.py) | ✓ | ✓ | ✓ |

`model.py`'yi hangi klasöre koyardık? Üçü de kullanıyor. Ayrıca proje genelinde
**56 yerde** `from target_point.X import ...` şeklinde çağrılıyorlar; klasörleri
bölmek bu çağrıların hepsini kırardı. Bu yüzden netliği fiziksel bölmeyle değil,
bu rehberle sağlıyoruz.
