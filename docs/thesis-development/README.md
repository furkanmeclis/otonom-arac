# Tez Geliştirme Dokümantasyonu

Bu klasör, `otonom-arac` projesinin Git geçmişi ve kod yapısından yeniden oluşturulan geliştirme süreci dokümantasyonunu içermektedir.

---

## Bu Dokümantasyonun Amacı

Bu belgeler, proje geliştirme sürecinde ayrı ayrı kaydedilmeyen teknik kararları, deneme izlerini, mimari evrimi ve sorun çözüm süreçlerini **Git geçmişi ve mevcut kod yapısından** yeniden belgelemek amacıyla oluşturulmuştur.

**Bu dokümantasyon:**
- Geliştirme sürecinin tez yazımı için kullanılabilecek akademik bir anlatısını sunar.
- Hangi bilgilerin kesin, hangilerinin tahmin olduğunu açıkça ayırt eder.
- Tez yazarının kendi bilgisiyle tamamlaması gereken boşlukları işaret eder.

**Bu dokümantasyon değildir:**
- Kesin tarihsel bir kayıt (commitlenmemiş denemeler kayıt dışı)
- Teknik doğruluk garantisi olan bir spesifikasyon
- Uydurulmuş veya büyütülmüş bir başarı anlatısı

---

## Kaynaklar

Bu belgeler aşağıdaki kaynaklardan üretilmiştir:

- `git log --oneline --reverse` — Commit kronolojisi
- `git show <commit>` — Her commit için değişen dosyalar
- `git log --stat` — Dosya bazında ekleme/silme istatistikleri
- Mevcut kaynak kod dosyaları (özellikle `model.py`, `controller.py`, `training.py`)
- `README.md` içeriği
- Dizin yapısı ve dosya isimlendirmeleri
- `data/artifacts/experiments/*/metrics.json` — Deney sonuçları
- `data/artifacts/reports/*/closed_loop_summary.json` — Değerlendirme raporları

---

## Dosya Listesi

| Dosya | İçerik |
|-------|--------|
| [00_proje_genel_ozet.md](00_proje_genel_ozet.md) | Projenin amacı, teknolojiler, sistem yapısı |
| [01_commit_kronolojisi.md](01_commit_kronolojisi.md) | Her commit için teknik analiz tablosu |
| [02_gelistirme_asamalari.md](02_gelistirme_asamalari.md) | Commitler anlamlı aşamalara gruplandırılmış |
| [03_deneyler_ve_denemeler.md](03_deneyler_ve_denemeler.md) | Silinen/değiştirilen kodlardan çıkarılan denemeler |
| [04_karsilasilan_problemler_ve_cozumler.md](04_karsilasilan_problemler_ve_cozumler.md) | Teknik problemler ve uygulanan çözümler |
| [05_teknik_kararlar.md](05_teknik_kararlar.md) | Mimari ve tasarım kararları |
| [06_sistem_mimarisi_evrimi.md](06_sistem_mimarisi_evrimi.md) | Dizin yapısının zaman içindeki değişimi |
| [07_ozellik_bazli_gelisim.md](07_ozellik_bazli_gelisim.md) | Her özelliğin bağımsız gelişim süreci |
| [08_hata_duzeltmeleri_ve_refactoring.md](08_hata_duzeltmeleri_ve_refactoring.md) | Hata düzeltmeleri ve kod düzenlemeleri |
| [09_test_dogrulama_ve_sinirlamalar.md](09_test_dogrulama_ve_sinirlamalar.md) | Test altyapısı ve proje sınırlamaları |
| [10_tezde_kullanilabilecek_anlatim.md](10_tezde_kullanilabilecek_anlatim.md) | Teze aktarılabilecek akademik paragraflar |
| [11_zaman_cizelgesi.md](11_zaman_cizelgesi.md) | Commit tarihlerine dayalı zaman çizelgesi |
| [12_eksik_bilgiler_ve_durust_notlar.md](12_eksik_bilgiler_ve_durust_notlar.md) | Belirsiz bilgiler, tahminler, tez yazarı için notlar |

---

## Tez Yazarken Nasıl Kullanılır?

### Başlangıç Noktaları

**Tezin "Sistem Tasarımı" bölümü için:**
→ [05_teknik_kararlar.md](05_teknik_kararlar.md) ve [06_sistem_mimarisi_evrimi.md](06_sistem_mimarisi_evrimi.md)

**Tezin "Yöntem" bölümü için:**
→ [07_ozellik_bazli_gelisim.md](07_ozellik_bazli_gelisim.md) ve [02_gelistirme_asamalari.md](02_gelistirme_asamalari.md)

**Tezin "Deneyler ve Değerlendirme" bölümü için:**
→ [03_deneyler_ve_denemeler.md](03_deneyler_ve_denemeler.md), [09_test_dogrulama_ve_sinirlamalar.md](09_test_dogrulama_ve_sinirlamalar.md) ve `data/artifacts/reports/` klasöründeki JSON raporları

**Hazır paragraflar için:**
→ [10_tezde_kullanilabilecek_anlatim.md](10_tezde_kullanilabilecek_anlatim.md)

**Tez öncesi kontrol için:**
→ [12_eksik_bilgiler_ve_durust_notlar.md](12_eksik_bilgiler_ve_durust_notlar.md) — mutlaka okuyun

---

## Kesin Bilgi ve Tahmin Ayrımı

Bu belgelerde tahmin içeren ifadeler şu kelimelerle işaretlenmiştir:
- "Muhtemelen"
- "Kod değişiminden anlaşıldığı kadarıyla"
- "Commit farkına göre"
- "Bu değişiklik şu amaca hizmet ediyor olabilir"
- "Kesin neden commit mesajında belirtilmemiştir"

Bu ifadelerle karşılaştığınızda teze yazmadan önce doğrulayın veya "tahmin" olduğunu belirtin.

---

## Güvenle Kullanılabilecek Bilgiler

Aşağıdakiler commit geçmişinden doğrudan doğrulanabilir bilgilerdir:

- Proje başlangıç tarihi: **2026-03-05**
- Son commit tarihi: **2026-04-26**
- Toplam commit sayısı: **39**
- Kullanılan teknolojiler: TensorFlow 2.15.1, Python 3.11, DonkeyCar, Unity/DonkeySim
- Target-point yaklaşımının benimsenmesi: **2026-03-24** (`6a8538a`)
- DAgger ve scripted expert denemelerinin kaldırılması: **2026-04-03** (`7caadc8`)
- `ai_pipeline/` yapısına geçiş: **2026-04-17** (`5de581c`)
- 9 model konfigürasyonunun tanımlanması: model_01 → model_12
