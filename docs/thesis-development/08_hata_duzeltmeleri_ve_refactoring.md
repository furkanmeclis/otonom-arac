# 08 — Hata Düzeltmeleri ve Refactoring

Bu dosyada commit geçmişinden tanımlanan hata düzeltmeleri, kod düzenlemeleri ve yapısal yeniden yapılanmalar belgelenmiştir.

---

## Değişiklik 1: Hardcoded Kişisel Yolların Kaldırılması

### Tür
Hata düzeltme / Kod temizliği

### İlgili Commitler
- `358d90e` — fix: scripts içindeki hardcoded kişisel yolları kaldır
- `d483ea6` — fix: update paths in documentation for user-specific directories

### Önceki Durum
PowerShell betiklerinde ve belgelerde sabit kullanıcı yolları bulunuyordu (örn. `C:\Users\alper\...`). Bu durum, reponun başka bir bilgisayarda ya da farklı kullanıcı adı altında çalıştırılmasını engelliyordu.

### Sonraki Durum
Sabit yollar dinamik değişkenler veya göreceli yollar ile değiştirildi. `simulationconfig.py` gibi yapılandırma dosyalarındaki yollar kullanıcı tarafından ayarlanabilir hale getirildi.

### Teknik Etki
Repo yeni bir ortamda klonlandığında doğrudan çalışabilir hale geldi.

### Tezde Kullanılabilecek Anlatım
"Proje, başlangıçta yalnızca geliştirici ortamında çalışacak şekilde yapılandırılmıştı. Yayına hazırlık aşamasında, hardcoded yollar genel parametrelerle değiştirilerek projenin farklı ortamlarda kullanılabilirliği sağlanmıştır."

---

## Değişiklik 2: Repo Klonlanabilirlik Düzeltmeleri

### Tür
Hata düzeltme

### İlgili Commitler
- `4ba0252` — fix: repo klonlanabilirliğini düzelt
- `7466d2b` — fix: yeni kullanıcıyı çökertecek 4 tutarsızlığı düzelt
- `fec3302` — fix: kalan küçük tutarsızlıkları gider

### Önceki Durum
Commit mesajı "yeni kullanıcıyı çökertecek 4 tutarsızlık" ifadesini içermektedir. Muhtemelen: eksik bağımlılık, yanlış yol referansı, eksik yapılandırma parametresi ve benzeri sorunlar.

### Sonraki Durum
Yeni kullanıcının kurulum ve çalıştırma sürecindeki hatalar giderildi.

### Teknik Etki
Repo dokümantasyonla birlikte başka bir sistemde çalıştırılabilir hale geldi.

---

## Değişiklik 3: Supervised Learning Örneklerinin Kaldırılması

### Tür
Refactoring / Kod temizliği

### İlgili Commitler
- `40d035d` — Refactor: Remove supervised learning scripts and configuration files

### Önceki Durum
`examples/supervised_learning/` dizininde standart DonkeyCar supervised learning yaklaşımına ait örnek kodlar bulunuyordu:
- `conf.py`, `evaluate.py`, `models.py`, `train.py`, `log/README.md`
- `models/example_model.h5`
- `test_cam_config.py`

### Sonraki Durum
Tüm bu dosyalar silindi. Projenin target-point yaklaşımına odaklandığı netleştirildi.

### Teknik Etki
Kod tabanı daha odaklı ve küçük hale geldi. Yeni geliştiriciler için hangi kodun geçerli olduğu konusundaki belirsizlik azaldı.

### Tezde Kullanılabilecek Anlatım
"Geliştirme sürecinde, projenin hedef nokta yaklaşımına tam geçişiyle birlikte, standart DonkeyCar supervised learning örnek kodları kaldırılmıştır. Bu adım, kod tabanının proje amacıyla tutarlılığını sağlamak için gerçekleştirilmiştir."

---

## Değişiklik 4: Target-Point Modülünün AI Pipeline'a Taşınması

### Tür
Refactoring / Yapısal düzenleme

### İlgili Commitler
- `5de581c` — Add new data artifacts, models, and documentation (bu committe taşıma gerçekleşti)

### Önceki Durum
```
otonom-arac/
├── target_point/          (kök dizinde)
├── build_target_point_labels.py
├── collect_target_point_data.py
└── evaluate_target_point.py
```

### Sonraki Durum
```
otonom-arac/
├── ai_pipeline/
│   ├── target_point/      (taşındı)
│   ├── build_target_point_labels.py
│   ├── collect_target_point_data.py
│   └── evaluate_target_point.py
```

### Teknik Etki
AI ile ilgili tüm bileşenler `ai_pipeline/` çatısı altında toplandı. Bu, yönetim betiği (`manage.py`) ve simülatör yapılandırmasından (`simulationconfig.py`) model kodunun ayrıştırılmasını sağladı.

### Tezde Kullanılabilecek Anlatım
"Kod tabanının büyümesiyle birlikte, ML pipeline bileşenleri araç kontrol altyapısından ayrıştırılarak `ai_pipeline/` dizini altında toplanmıştır. Bu yapısal düzenleme, ilgili bileşenlerin bir arada bulunmasını (cohesion) artırırken farklı sorumluluk alanları arasındaki bağlantıyı azaltmıştır (loose coupling)."

---

## Değişiklik 5: Train.py'nin Cihaz Yapılandırmasıyla Güncellenmesi

### Tür
Özellik / Refactoring

### İlgili Commitler
- `c5b68e3` — Enhance training script with device configuration

### Önceki Durum
`train.py` sabit donanım yapılandırmasıyla çalışıyordu.

### Sonraki Durum
Eğitim betiğine GPU/CPU seçimi ve cihaz yapılandırması eklendi. `requirements-train.txt` ayrı bir dosya olarak oluşturuldu.

### Tezde Kullanılabilecek Anlatım
"Farklı donanım ortamlarında çalışabilirliği sağlamak amacıyla eğitim betiği cihaz yapılandırma desteğiyle genişletilmiştir."

---

## Değişiklik 6: DAgger ve İlgili Bileşenlerin Kaldırılması (Büyük Revert)

### Tür
Refactoring / Deneme kaldırma

### İlgili Commitler
- `7caadc8` — revert: roll back to phase5 adaptive robust baseline

### Önceki Durum
`target_point/` modülü şu ek bileşenleri içeriyordu:
- `dagger.py` (DAgger algoritması)
- `scripted_expert.py` (kural tabanlı uzman)
- `promotion.py` (model promosyon mekanizması)
- `temporal.py` (zamansal model)
- `effective_loss.py` (özel kayıp fonksiyonu)

`training.py` ~757 satırdı (karmaşık hali).
`tests/test_target_point.py` ~989 satırdı (kapsamlı test seti).

### Sonraki Durum
Bahsi geçen dosyalar kaldırıldı. `training.py` sadeleştirildi.

### Teknik Etki
165 dosyada değişiklik; 4829 satır silindi. Sistem daha az karmaşık ama daha sağlam bir baseline'a döndü.

---

## Değişiklik 7: README'nin Yeniden Yazılması

### Tür
Dokümantasyon / Refactoring

### İlgili Commitler
- `b27709b` — refactor: update README to reflect project structure

### Önceki Durum
README muhtemelen eski dizin yapısını veya standart DonkeyCar talimatlarını yansıtıyordu.

### Sonraki Durum
README yeniden yazılarak:
- Projenin amacı (target-point sim2real)
- Hızlı başlangıç talimatları
- Güncel mimari özeti
- Proje yapısı

netleştirildi.

---

## Değişiklik 8: Controller'a Bias ve Deadband Eklenmesi

### Tür
Hata düzeltme / İnce ayar

### İlgili Commitler
- `2995dab` — Add target-point bias compensation and deadband parameters

### Önceki Durum
Kontrolcü saf geometrik hesaplama yapıyordu; donanım kaynaklı sistematik sapma dikkate alınmıyordu.

### Sonraki Durum
Bias kompansasyonu ve deadband parametreleri eklendi. Bu parametreler `simulationconfig.py` üzerinden ayarlanabilir hale getirildi.

### Teknik Etki
Gerçek araç sürüşünde direksiyon tutarlılığı ve istikrarı arttı.

---

## Değişiklik 9: Batch Size Optimizasyonu

### Tür
Performans iyileştirme

### İlgili Commitler
- `ec10b31` — feat: Update simulation configurations

### Önceki Durum
`TARGET_POINT_BATCH_SIZE = 32`

### Sonraki Durum
`TARGET_POINT_BATCH_SIZE = 128`

### Teknik Etki
Eğitim süresi önemli ölçüde kısaldı. Büyük veri kümelerinde GPU kullanım verimliliği arttı.
