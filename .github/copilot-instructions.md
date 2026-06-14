# AI Agent First-Read Guide (otonom-arac)

Bu dosya proje icin ana AI ogretim dosyasidir.
Bu depoda gorev alan her AI ajan su sirayi izlemelidir:
1. Once bu dosyayi tam oku.
2. Sonra istenen gorevle ilgili alt bolume git.
3. Gerekli dosyalari ac, degisikligi yap, test et, raporla.

## 1) Proje amaci
Bu repo iki ana parcayi birlestirir:
- gym_donkeycar: Donkey Car icin OpenAI Gym simulasyon ortami.
- ai_pipeline/target_point: Target-point tabanli otonom surus egitim/veri/evaluasyon hatti.

Hedef:
- Jetson Nano uzerinde calisan otonom aracin tum simulasyon haritalarinda stabil ve tekrar edilebilir surus yapmasi.
- Jetson Nano ile gercek hayattaki parkurlarda da parkur disina cikmadan dogru surus yapmasi.
- Sim -> real gecisinde preprocess, model cikisi ve controller davranisini birebir uyumlu tutmak.
- Kapali dongu metriklerle (completion, offtrack, recovery, TTF) ve saha testleriyle kaliteyi izlemek.

## 2) Isletim kurali (ajan davranis sozlesmesi)
- Bu dosyadaki isimleri ve asama akisini referans almadan kod degistirme.
- Veri boru hattinda asama sirasini bozma: map -> collect -> manifest -> train -> evaluate.
- Manifest acik verilmeden egitim baslatma.
- simulationconfig.py ile train/runtime preprocess uyumunu koru.
- Her degisiklikte ana hedefe etkisini degerlendir: tum haritalarda stabil surus + gercek parkurda pist disina cikmama.
- Sadece simulasyonda iyi gorunen degisiklikleri tamamlanmis sayma; Jetson Nano saha dogrulamasini planla.
- data/, donkey_sim.app/, gym_donkeycar.egg-info/ altinda elle duzenleme yapma; bunlar veri/binary/uretilmis artefakt olabilir.

## 3) Uc ana sistem

### A) Simulasyon cekirdegi
- gym_donkeycar/core + gym_donkeycar/envs
- Unity simulator ile TCP haberlesme, env adimlama, reward/done/info uretimi.

### B) AI pipeline (target-point)
- ai_pipeline/collect_target_point_data.py
- ai_pipeline/build_target_point_labels.py
- ai_pipeline/train.py
- ai_pipeline/evaluate_target_point.py
- ai_pipeline/target_point/*

### C) Runtime surus
- manage.py -> target_point pilot + controller zinciri
- modelden (target_x, target_y) cikar, direksiyon/gaz komutuna donustur.
- Jetson Nano hedefinde .keras/.tflite model ile ayni surus mantigini koruyacak sekilde calistirilir.

## 4) End-to-end asamalar (kritik)

### Phase 1: Mapping
- Giris: collect_target_point_data.py --task map
- Nasil: clean teacher ile pist izi toplanir, centerline ve map metadata uretilir.
- Cikti: artifacts/maps/... altinda map artefaktlari.

### Phase 2: Dusuk gurultu veri toplama
- Giris: collect_target_point_data.py --task collect --collection-profile phase2_low_noise
- Nasil: dusuk gurultulu ogretmen verisi uretilir.
- Cikti: data/target_point_phase2*/raw

### Phase 3: Yuksek gurultu + recovery veri toplama
- Giris: collect_target_point_data.py --task collect --collection-profile phase3_full_noise
- Nasil: domain randomization ve recovery senaryolari eklenir.
- Cikti: data/target_point_phase3*/raw

### Phase 4: Manifest/label materyalizasyon
- Giris: build_target_point_labels.py --raw-root/--raw-roots
- Nasil: filtreleme, dengeleme, dual-label modlari (fixed_1p2m/adaptive_v1).
- Cikti: index altinda jsonl/csv/raporlar.

### Phase 5: Egitim
- Giris: train.py --type target_point --manifest ...
- Nasil: manifest-native veri yukleme, agirlikli loss, hard-example/diagnostics.
- Cikti: models/*.keras ve deney raporlari.

### Phase 6: Kapali dongu degerlendirme
- Giris: evaluate_target_point.py --model ... --tracks ...
- Nasil: pist bazli episode kosulur, completion/offtrack/recovery/TTF raporlanir.
- Cikti: artifacts/target_point/reports/*

### Opsiyonel Phase 5.5: Rollout bootstrap
- Giris: collect_target_point_data.py --task rollout_collect --driver-model ...
- Nasil: modelin zorlandigi sahnelerden recovery/failure-margin veri toplar.
- Cikti: rollout raw + ozet raporlar.

### Phase 7: Jetson Nano gercek parkur dogrulama
- Giris: manage.py drive --type target_point --model ... --js
- Nasil: simulasyondaki preprocess/controller parametreleri ile birebir parity korunarak saha testi yapilir.
- Cikti: saha test notlari, lap stabilitesi, manuel mudahale/offtrack kaydi.

## 5) Dosya katalogu (ne yapar / nasil yapar)

### 5.1 Kok dizin dosyalari
- README.md: Projenin genel tanimi. Nasil: gym_donkeycar kurulum ve ornek env kullanimini anlatir.
- AUTHORS.rst: Katki verenler listesi. Nasil: dokumantasyon metadata.
- CONTRIBUTING.rst: Katki kurallari. Nasil: gelistirme sureci ve beklentiler.
- HISTORY.rst: Surum gecmisi. Nasil: degisiklik logu.
- LICENSE: Lisans metni. Nasil: MIT lisans kosullari.
- Makefile: Gelistirme komutlari. Nasil: type/lint/format/test/docs hedefleri.
- MANIFEST.in: Paket dagitim icerigi. Nasil: hangi dosyalar wheel/sdist'e dahil edilir.
- pyproject.toml: Tool ayarlari. Nasil: black, ruff, isort, pytype konfigurasyonu.
- setup.cfg: Ek paket/pytest ayarlari. Nasil: pytest collect_ignore vb.
- setup.py: Paket kurulum tarifi. Nasil: setuptools ile dependency ve extras tanimi.
- requirements-train.txt: Egitim ortam bagimliliklari. Nasil: donkeycar, tensorflow, gym, numpy, pillow, docopt vb.
- config.py: DonkeyCar varsayilan arac konfigurasyonu. Nasil: donanim/surus/kamera drivetrain ayarlari.
- simulationconfig.py: Simulasyon ve target-point merkez config. Nasil: TARGET_POINT_* hiperparametreleri ve veri yolaklari.
- manage.py: Ana arac komutlari (drive/train/smoke). Nasil: DonkeyCar parcalarini pipeline olarak birlestirir, target_point pilot/controller entegre eder.
- calibrate.py: Kalibrasyon yardimcisi. Nasil: drive train/parca ayarlarini test etmek icin minimal arac dongusu.

### 5.2 .github
- .github/ISSUE_TEMPLATE.md: Issue sablonu. Nasil: bug raporlama formatini standartlastirir.
- .github/workflows/ci.yml: CI akisi. Nasil: matrix python surumlerinde type/codestyle/lint adimlarini calistirir.
- .github/copilot-instructions.md: Bu dosya. Nasil: AI ajanlar icin birinci kaynak talimat.

### 5.3 ai_pipeline giris scriptleri
- ai_pipeline/collect_target_point_data.py: Mapping + dataset collection entrypoint. Nasil: --task map/collect/rollout_collect ile collector/mapping fonksiyonlarini cagirir.
- ai_pipeline/build_target_point_labels.py: Label/manifest olusturma girisi. Nasil: map label ya da raw->index manifest islemlerini target_point.manifest ve teacher_policy ile yapar.
- ai_pipeline/train.py: Egitim girisi. Nasil: device secimi (cpu/gpu/auto), --type target_point oldugunda target_point.training.train_target_point cagirir.
- ai_pipeline/evaluate_target_point.py: Kapali dongu evaluasyon girisi. Nasil: evaluate_closed_loop fonksiyonunu config override parametreleriyle cagirir.

### 5.4 ai_pipeline/target_point modulleri
- ai_pipeline/target_point/__init__.py: Lazy import facade. Nasil: __getattr__ ile agir modulleri ihtiyac halinde yukler.
- ai_pipeline/target_point/model.py: Model ve preprocess. Nasil: crop/resize/normalize, efficient veya legacy ag, denormalizer katmani.
- ai_pipeline/target_point/pilot.py: Runtime model inferencer. Nasil: modeli yukler, frame alir, target_x/target_y cikarir.
- ai_pipeline/target_point/pilot_tflite.py: TFLite inferencer. Nasil: tflite interpreter ile hizli edge tahmin zinciri.
- ai_pipeline/target_point/controller.py: Target point -> kontrol. Nasil: geometri tabanli steering/throttle hesaplar, min-forward ve rate limit korumalari uygular.
- ai_pipeline/target_point/dataset.py: Dataset ve manifest yukleme. Nasil: tub/manifests parse eder, TargetPointSample uretir, split/karisim yardimcilari sunar.
- ai_pipeline/target_point/training.py: Manifest-native egitim. Nasil: normalization istatistikleri, sample weighting, hard-example stratejisi, keras sequence/loss/callback/rapor uretimi.
- ai_pipeline/target_point/diagnostics.py: Tanilama ve analiz. Nasil: prediction summary, segment metrikleri, collapse gate, contact sheet/rapor yazimi.
- ai_pipeline/target_point/augment.py: Deterministic augmentation. Nasil: clip-consistent goruntu donusumleri uygular.
- ai_pipeline/target_point/domain_randomization.py: Episode bazli domain randomization. Nasil: deterministic profile sample edip goruntuye uygular.
- ai_pipeline/target_point/teacher_policy.py: Ogretmen politikasi ve etiketleme modlari. Nasil: fixed/adaptive lookahead, map/collect icin teacher hedefleri.
- ai_pipeline/target_point/collector.py: Raw veri toplayici. Nasil: sim session + teacher + map kullanarak episode satirlari/imageleri kaydeder; rollout toplama da burada.
- ai_pipeline/target_point/mapping.py: Faz-1 map yardimcilari. Nasil: clean trace toplayip track_map uretilmesini yonetir.
- ai_pipeline/target_point/track_map.py: Geometri ve map artefaktlari. Nasil: centerline waypoint yapilari, curvature, mesafe ve save/load islemleri.
- ai_pipeline/target_point/manifest.py: Raw->index filtre/dengeleme. Nasil: jsonl okuma-yazma, kalite filtreleri, recovery/rollout cap, dual label manifesleri.
- ai_pipeline/target_point/evaluate_closed_loop.py: Closed-loop evaluasyon motoru. Nasil: episode bazli metrik hesaplar, track aggregate ve rapor cikarir.
- ai_pipeline/target_point/export.py: Model export araci. Nasil: Keras modelini INT8/FP16/FP32 TFLite formatlarina cevirir, benchmark yardimcilari saglar.
- ai_pipeline/target_point/experiments.py: Deney klasoru/rapor yardimcilari. Nasil: deney dizini hazirlar, json payload yazdirir.
- ai_pipeline/target_point/external_adapter.py: Harici dataset donusturucu. Nasil: dis tub veri formatini target-point sample/manifeste map eder.
- ai_pipeline/target_point/sim_session.py: Deterministic simulasyon oturumu. Nasil: gym env ac/kapat, reset/step, observation sarmalama.

### 5.5 ai_pipeline/tools
- ai_pipeline/tools/: Su an bos. Nasil: gelecekte pipeline yardimci scriptleri icin ayrilan klasor.

### 5.6 gym_donkeycar paketi
- gym_donkeycar/__init__.py: Environment register noktasi. Nasil: gym register ile donkey-* env ID'lerini yayinlar.
- gym_donkeycar/version.txt: Paket surumu. Nasil: __init__.py tarafindan okunur.

#### gym_donkeycar/core
- gym_donkeycar/core/__init__.py: Core paket isaretleyici.
- gym_donkeycar/core/client.py: TCP istemci altyapisi. Nasil: SDClient async socket loop ve message framing saglar.
- gym_donkeycar/core/sim_client.py: Sim istemci uyarlamasi. Nasil: SDClient'i simulator mesaj formatina uyarlar.
- gym_donkeycar/core/message.py: Mesaj handler arabirimi. Nasil: IMesgHandler soyut metodlarini tanimlar.
- gym_donkeycar/core/fps.py: FPS olcer. Nasil: frame sayaci ile hiz loglar.
- gym_donkeycar/core/util.py: JSON float notation yardimcisi. Nasil: locale kaynakli ondalik bicimlerini normalize eder.

#### gym_donkeycar/envs
- gym_donkeycar/envs/__init__.py: Env paket isaretleyici.
- gym_donkeycar/envs/donkey_env.py: Gym Env ana sinifi. Nasil: action/obs spaces, reset/step, reward/done/info akisini yonetir.
- gym_donkeycar/envs/donkey_sim.py: Unity simulator kontrolu. Nasil: sim handler/controller, quaternion ve telemetri islemleri.
- gym_donkeycar/envs/donkey_proc.py: Simulator process yonetimi. Nasil: unity process spawn/stop.
- gym_donkeycar/envs/donkey_ex.py: Ortam ozel exception'lari. Nasil: SimFailed vb durumlari isaretler.

#### gym_donkeycar/test
- gym_donkeycar/test/README.rst: Manuel socket test aciklamasi.
- gym_donkeycar/test/client.test.py: TCP message parser regression testi. Nasil: local echo server kurup parcali payload senaryolari dener.

### 5.7 tests
- tests/conftest.py: Test import yolu kurulumu. Nasil: root ve ai_pipeline path ekler.
- tests/test_gym_donkeycar.py: Env register smoke testi. Nasil: tum env id'lerini gym.make ile dogrular.
- tests/test_target_point.py: Target-point unit testleri. Nasil: compute_target_point, controller min-forward, tub path ve collapse gate davranislarini assert eder.
- tests/core/test_fps.py: FPSTimer testi. Nasil: on_frame/reset davranisini kontrol eder.

### 5.8 docs
- docs/index.rst: Sphinx ana index.
- docs/installation.rst: Kurulum adimlari.
- docs/usage.rst: Kullanim ornekleri.
- docs/gym_donkeycar.rst: Paket API dokumu.
- docs/gym_donkeycar.core.rst: Core API dokumu.
- docs/gym_donkeycar.envs.rst: Env API dokumu.
- docs/modules.rst: Sphinx modul listesi.
- docs/authors.rst: Yazarlar sayfasi.
- docs/contributing.rst: Katki sayfasi.
- docs/history.rst: Gecmis/surum notlari.
- docs/conf.py: Sphinx konfig dosyasi.
- docs/Makefile: Unix docs build komutlari.
- docs/make.bat: Windows docs build komutlari.
- docs/SETUP.md: Projeye ozel hizli kurulum rehberi (Python 3.11, venv, smoke test).
- docs/COMMANDS.md: Projeye ozel komut akisi (map/collect/manifest/train/evaluate).
- docs/Instruction.md: Cok asamali uygulama plani ve pass/fail kapilari.
- docs/_static/: Sphinx statik varliklari.

### 5.9 Buyuk/uretilmis varlik klasorleri
- data/: Dataset ve artefakt deposu. Nasil: raw koleksiyon, index manifest, raporlar ve harici datasetler burada tutulur.
- data/datasets/donkey_datasets/: Ham donut setleri.
- data/datasets/extracted/: Harici dataset ciktilari.
- donkey_sim.app/: macOS simulator binary paketi. Elle duzenlenmez.
- gym_donkeycar.egg-info/: Paket metadata ciktilari (setup/install kaynakli). Elle duzenlenmez.
- __pycache__/ klasorleri: Python cache, yok sayilir.

## 6) Config odak noktasi
Target-point ile ilgili neredeyse tum calisma parametreleri simulationconfig.py icindedir:
- Model/preprocess: TARGET_POINT_IMAGE_*, TARGET_POINT_CROP_*, TARGET_POINT_MODEL_ARCH
- Egitim: TARGET_POINT_BATCH_SIZE, TARGET_POINT_MAX_EPOCHS, TARGET_POINT_LOSS_*, TARGET_POINT_TARGET_MIN_STD
- Ornek agirliklari/hard-case: TARGET_POINT_*_SAMPLE_WEIGHT, TARGET_POINT_HARD_*
- Runtime kontrol: TARGET_POINT_STEER_GAIN, TARGET_POINT_DYNAMIC_THROTTLE, TARGET_POINT_STEER_RATE_LIMIT, TARGET_POINT_ANTICIPATION_*
- Evaluasyon: TARGET_POINT_EVAL_*
- Bootstrap/rollout: TARGET_POINT_BOOTSTRAP_*

Ajan bir degisiklikte config parity kontrolu yapmalidir:
- Train preprocess ve runtime preprocess zinciri ayni mi?
- Egitim manifesti beklenen index'e mi bakiyor?
- Evaluasyon seed ve throttle parametreleri raporda belirtilmis mi?

## 7) SIK gorev -> hangi dosyaya gidilir
- Veri topla: ai_pipeline/collect_target_point_data.py, ai_pipeline/target_point/collector.py, ai_pipeline/target_point/teacher_policy.py
- Label/manifest: ai_pipeline/build_target_point_labels.py, ai_pipeline/target_point/manifest.py
- Egitim: ai_pipeline/train.py, ai_pipeline/target_point/training.py, ai_pipeline/target_point/model.py
- Runtime surus: manage.py, ai_pipeline/target_point/pilot.py, ai_pipeline/target_point/controller.py
- Closed-loop degerlendirme: ai_pipeline/evaluate_target_point.py, ai_pipeline/target_point/evaluate_closed_loop.py
- Sim env sorunu: gym_donkeycar/envs/donkey_env.py, gym_donkeycar/envs/donkey_sim.py, gym_donkeycar/core/*
- Test kirilmasi: tests/test_target_point.py, tests/test_gym_donkeycar.py, tests/core/test_fps.py

## 8) Kalite kapilari (onerilen minimum)
- Lint/format/type:
  - make type
  - make check-codestyle
  - make lint
- Test:
  - pytest -v tests/
- Pipeline degisikliginde ek kontrol:
  - Tum hedef haritalarda closed-loop evaluate calistir.
  - Raporlarda completion/offtrack/TTF degisimini not et.
  - Ana hedef metrikleri icin sim kabulunu kontrol et: parkur disina cikma olmamasi (offtrack=0 hedefi) ve stabil lap tekrari.
  - Jetson Nano gercek parkur testi olmadan gorevi tam kapanmis sayma.

## 9) Ajan cevap formati onerisi
Ajan bir istegi tamamlarken su sirayi izlesin:
1. Hangi asamayi etkiledigini yaz (map/collect/manifest/train/evaluate/runtime).
2. Hangi dosyalari degistirdigini yaz.
3. Neden bu mekanizmayi sectigini yaz.
4. Neyi calistirip nasil dogruladigini yaz.
5. Risk/varsayim varsa acik belirt.

Bu dosya proje icin birincil AI onboard kaynagidir.
