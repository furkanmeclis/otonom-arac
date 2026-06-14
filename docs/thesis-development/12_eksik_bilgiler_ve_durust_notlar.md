# 12 — Eksik Bilgiler ve Dürüst Notlar

Bu dosya, Git geçmişinden net olarak çıkarılamayan bilgileri ve analiz boyunca yapılan tahminleri derlemektedir. Tez yazarken bu dosyayı dikkatle okuyun: hangi bilgilerin kesin, hangilerinin tahmin olduğunu buradan teyit edin.

---

## Net Olarak Çıkarılamayan Bilgiler

### 1. Deneysel Sonuçların Karşılaştırması

**Sorun:** Repo'da birçok `metrics.json` ve `closed_loop_summary.json` dosyası mevcut; ancak hangi modelin "en iyi" olduğuna dair kesin bir sonuç belgesi ya da commit mesajı bulunmamaktadır.

**Ne bilinmiyor:** Model 01'den 12'ye kadar olan konfigürasyonların karşılaştırmalı sonuçları; hangi veri stratejisinin (pure sim, hybrid, pure real) hangi pistlerde üstün performans gösterdiği.

**Tez için:** Bu sonuçları `metrics.json` dosyalarından kendiniz okuyarak belgeleyin.

---

### 2. Gerçek Araç Üzerinde Test

**Sorun:** TFLite pilot kodu (`pilot_tflite.py`) yazılmış ve Jetson için optimize edilmiş. Ancak commit geçmişinde gerçek araç üzerinde test yapıldığına dair açık bir iz yok.

**Ne bilinmiyor:** Modelin gerçek bir araçta test edilip edilmediği; edildi ise sonuçlar ne oldu.

**Tez için:** Gerçek araç testleri yaptıysanız bu bilgiyi teze kendiniz ekleyin.

---

### 3. DAgger ve İleri Tekniklerin Başarısızlık Nedeni

**Sorun:** `7caadc8` reverte DAgger, scripted expert ve diğer bileşenler kaldırıldı; ancak commit mesajı sadece "phase5 adaptive robust baseline'a geri dönüş" yazmakta.

**Ne bilinmiyor:** Bu tekniklerin tam olarak neden başarısız ya da yetersiz bulunduğu. Metrik başarısızlığı mı, kararlılık sorunu mu, zaman kısıtı mı?

**Bu analiz ne diyor:** "Muhtemelen bu teknikler beklenen performansı sağlamadı veya sistemin yönetimi zorlaştı." Bu bir tahmindir, kesin değildir.

---

### 4. Commitlenmemiş Denemeler

**Sorun:** Git geçmişi yalnızca commitlenmiş değişiklikleri gösterir. Yerel olarak denenen ancak commitlenmeyen:
- Başarısız model mimarileri
- Test edilmemiş hiperparametre kombinasyonları
- Yarıda bırakılan özellik geliştirmeleri
- Manuel sürüş denemeleri

bunların hiçbiri git geçmişinde görünmemektedir.

**Tez için:** Hatırladığınız önemli yerel denemeleri "12_eksik_bilgiler" bölümüne kendiniz ekleyin.

---

### 5. Phase Numaralandırması Mantığı

**Sorun:** Commit mesajlarında "phase 4", "phase 5", "phase 5.5" gibi numaralar geçmektedir; ancak bu numaraların neyi temsil ettiği açıkça belirtilmemiştir.

**Ne bilinmiyor:** Phase 1, Phase 2, Phase 3 neredeydi? Bu numaralandırma sistematik bir plan mı yoksa geliştirme sırasında ortaya mı çıktı?

**Bu analiz ne diyor:** Phase numaraları muhtemelen eğitim deney aşamalarına karşılık gelmektedir; ancak tüm aşamaların commit geçmişinde temsil edilip edilmediği belirsizdir.

---

### 6. "last" ve "Son yapılan alan" Commit Mesajları

**Sorun:** Commit `2952627` ("last") ve `41389c3` ("Son yapılan alan") belirsiz mesajlar içermektedir. Diff incelemesi yapılmadan içerikleri belirlenemez.

**Ne bilinmiyor:** Bu commitlerde tam olarak ne değiştirildi.

---

### 7. Model Performans Beklentileri

**Sorun:** Hangi performans seviyesinin (kaç tur tamamlama oranı, kaç pist dışı çıkma) "yeterli" sayıldığı commit mesajlarında belirtilmemiştir.

**Ne bilinmiyor:** Başarı kriterleri. Örneğin: "5 turdan 4'ünü tamamlamak başarı sayılır" gibi bir eşik tanımlanmış mı?

---

### 8. Eğitim Süreleri

**Sorun:** Commit geçmişinde eğitim sürelerine dair bilgi bulunmamaktadır.

**Ne bilinmiyor:** Tek bir modelin eğitilmesi ne kadar sürdü? Toplam GPU saati ne kadardı? Bu bilgi, büyük ölçekli veri toplama betiklerinin yazılmasıyla dolaylı olarak tahmin edilebilir (uzun sürelerin otomasyon gerektirdiği anlaşılmaktadır).

---

### 9. Hangi Pistlerde Test Yapıldığı

**Sorun:** 6 pist haritalanmış; ancak model değerlendirmesinin hangi pistlerde yapıldığı ve sonuçların pistten piste nasıl değiştiği net değildir.

---

### 10. Myconfig Versiyon Geçmişi

**Sorun:** `myconfig.py` defalarca güncellendi; ancak her versiyonun tam parametrelerini görmek için tüm diff geçmişinin incelenmesi gerekir.

---

## Tahmine Dayalı Yorumlar

Aşağıdaki ifadeler bu analizde kullanılmış tahminlerdir. Teze aktarırken bunların tahmin olduğunu belirtin veya doğrulayın:

| Tahmin | Dayanağı | Kesinlik Düzeyi |
|--------|----------|----------------|
| DAgger başarısız bulundu | Revert commit + kaldırılan modüller | Orta |
| Scripted expert gerçek veriden kalite olarak geride kaldı | Modül silindi | Düşük |
| Batch size artışı eğitim süresini kısalttı | 32→128 değişikliği, zamanla ilgili yorum yok | Düşük |
| Phase numaraları eğitim döngülerini temsil ediyor | İsimlendirme kalıbı | Orta |
| macOS'ta geliştirme yapılıp Windows'a taşındı | `__MACOSX/` artefaktları | Yüksek |
| Gerçek araç Jetson kullanıyor | `pilot_tflite.py` + Jetson referansları | Yüksek |

---

## Tez Yazarken Dikkat Edilmesi Gerekenler

### Abartmaktan Kaçının
- "DAgger başarısız oldu" değil, "DAgger kaldırıldı; kesin neden belgelenmiş değil"
- "Model X en iyi sonucu verdi" değil, metrik dosyalarından okunan değerleri yazın
- "Sistem gerçek araçta başarıyla çalıştı" — bunu yalnızca gerçekten test yaptıysanız yazın

### Mutlaka Kendinizin Tamamlaması Gereken Kısımlar
1. Deneysel sonuçların karşılaştırmalı tablosu (metrics.json'lardan)
2. Gerçek araç test sonuçları (varsa)
3. Neden belirli tekniklerin terk edildiğine dair kişisel notlar
4. Eğitim süreleri
5. Başarı kriterleri tanımı

### Güvenle Kullanabileceğiniz Bilgiler
Aşağıdakiler commit geçmişinden kesin olarak doğrulanmıştır:
- Proje tarihleri (2026-03-05 başlangıç, 2026-04-26 son commit)
- Kullanılan teknoloji yığını (TF 2.15.1, DonkeyCar, Unity)
- Target-point yaklaşımının benimsenmesi
- Phase 4 ve Phase 5 deneylerinin yapıldığı
- DAgger ve scripted expert modüllerinin oluşturulup kaldırıldığı
- `ai_pipeline/` yapısına geçişin gerçekleştiği
- 9 model konfigürasyonunun tanımlandığı
- Kapalı döngü değerlendirme raporlarının üretildiği

---

## Öğrencinin Sonradan Ekleyebileceği Bilgiler

Aşağıdaki bölümler için sizi yönlendirecek sorular:

**Başlangıç Motivasyonu:**
- Neden DonkeyCar çerçevesini seçtiniz?
- Hedef nokta yaklaşımını hangi kaynaktan öğrendiniz?

**Deneysel Süreç:**
- DAgger denemesi ne kadar sürdü ve neden kaldırdınız?
- En iyi performansı hangi model konfigürasyonu verdi?
- Phase numaraları neyi temsil ediyor?

**Gerçek Araç:**
- Jetson üzerinde test yaptınız mı? Sonuç ne oldu?
- Simülasyon ve gerçek dünya arasındaki gözlemlediğiniz en büyük fark neydi?

**Öğrendikleriniz:**
- Sonradan farklı yapardınız dediğiniz bir şey var mı?
- Tezin katkısı olarak öne çıkarmak istediğiniz tek bir sonuç nedir?
