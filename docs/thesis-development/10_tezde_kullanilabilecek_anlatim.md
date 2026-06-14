# 10 — Tezde Kullanılabilecek Geliştirme Süreci Anlatımı

Bu dosya, teze doğrudan aktarılabilecek ya da uyarlanabilecek paragraflar içermektedir. Tüm ifadeler commit geçmişi ve kod incelemesine dayandırılmıştır; tahmin içeren kısımlar açıkça işaretlenmiştir.

---

## Projenin Başlangıcı

Bu çalışmada otonom araç kontrol sistemi geliştirilmesi için DonkeyCar açık kaynak çerçevesi temel alınmıştır. DonkeyCar, düşük maliyetli otonom araç projeleri için araç kontrolü, simülatör bağlantısı ve temel veri toplama altyapısı sunan kapsamlı bir çerçevedir. Projenin başlangıç aşamasında çerçeve yerel ortama kurulmuş, temel eğitim altyapısı doğrulanmış ve ilk model denemeleri gerçekleştirilmiştir.

Simülasyon ortamı olarak Unity tabanlı DonkeySim kullanılmıştır. Bu simülatör, aracın çeşitli sanal pistlerde sürülmesine, görüntü verisi toplanmasına ve farklı ışık ile fizik koşullarının test edilmesine olanak tanımaktadır.

---

## Hedef Nokta Yaklaşımının Benimsenmesi

Projenin temel mimari kararı, doğrudan direksiyon açısı tahmiri yerine hedef nokta (target-point) tahmirinin benimsenmesidir. Bu yaklaşımda model, kamera görüntüsünden ego-frame koordinat sisteminde ilerleyen bir noktanın `(target_x, target_y)` koordinatını tahmin eder. Tahmin edilen koordinat, geometrik bir kontrolcü tarafından direksiyon ve gaz komutlarına dönüştürülmektedir.

Bu mimari tercihin temel motivasyonu, simülasyon-gerçek transfer (sim-to-real) performansıdır. Doğrudan direksiyon tahmiri, simülasyon ile gerçek dünya arasındaki görsel farklılıklara (domain gap) duyarlı iken, hedef nokta tahmiri bu problemi görsel olmayan bir geometrik ara temsile taşıyarak daha sağlam bir transfer sağlamaktadır.

Hedef nokta tahminine dayalı kontrol mimarisi üç ana bileşenden oluşmaktadır: görüntüden koordinat tahmin eden CNN modeli, koordinattan direksiyon/gaz komutu hesaplayan geometrik kontrolcü ve eğitim verisi için pist merkez hattı haritalayan etiketleme pipeline'ı.

---

## İlk Prototipin Oluşturulması

Hedef nokta modülü (`target_point/`) üç hafta süren geliştirme sürecinin ardından sisteme entegre edilmiştir. Bu modül; model, kontrolcü, veri kümesi yükleme, eğitim döngüsü, teşhis araçları ve gerçek zamanlı çıkarım bileşenlerini kapsamaktadır.

Model mimarisi, ~115.000 parametre içeren ve depthwise separable evrişim katmanları kullanan bir CNN olarak tasarlanmıştır. Bu mimari seçimi, parametresi daha az olan ancak benzer ifade gücüne sahip modeller oluşturmayı hedefleyen verimli ağ tasarım ilkelerini yansıtmaktadır. Gömülü donanımda (Jetson platformu) çalışabilirlik hedefi göz önünde bulundurulduğunda, parametre etkinliği önemli bir tasarım kısıtı olarak belirlenmiştir.

---

## Pist Haritalama ve Etiketleme

Hedef nokta tabanlı eğitim, her pist için merkez hat bilgisi gerektirmektedir. Bu amaçla iki aşamalı bir etiketleme pipeline'ı geliştirilmiştir.

İlk aşamada araç simülatörde pist boyunca sürülmekte ve konum verileri `raw_trace.csv` formatında kaydedilmektedir. İkinci aşamada ham iz verisi işlenerek pist merkez hattı (`centerline.csv`) hesaplanmakta ve her görüntü karesi için uygun hedef nokta koordinatı geometrik yöntemle belirlenmektedir.

Hedef nokta seçiminde iki farklı strateji karşılaştırmalı olarak geliştirilmiştir: 1.2 metre sabit ileri bakış mesafesi (fixed lookahead) ve pist eğriliğine duyarlı adaptif ileri bakış mesafesi (adaptive lookahead). Bu sistematik karşılaştırma, projenin 6 farklı DonkeyCar pistini kapsayan kapsamlı bir deneysel altyapıya dayanmaktadır.

---

## Temel Özelliklerin Geliştirilmesi

İlk çalışan prototipten sonra sistem çok sayıda ek bileşenle genişletilmiştir. Veri artırma (`augment.py`) ile eğitim sırasında görüntü dönüşümleri uygulanmıştır; yatay çevirme gibi temel dönüşümler, pist simetrisinden yararlanarak eğitim verisi etkin biçimde katlanmıştır.

Domain randomizasyon (`domain_randomization.py`) ile simülasyon görüntüleri, gerçek dünya görüntülerinin çeşitliliğini daha iyi yansıtacak biçimde dönüştürülmüştür. Bu teknik, modelin yalnızca simülasyona özgü görsel ipuçlarına aşırı uyum sağlamasını (overfitting) önlemeyi hedeflemektedir.

Kapalı döngü değerlendirme çerçevesi (`evaluate_closed_loop.py`), modelin simülatörde tam otonom sürüş yapmasını ve performans metriklerinin (tamamlanan tur sayısı, pist dışına çıkma sıklığı) otomatik olarak kaydedilmesini sağlamıştır.

---

## Karşılaşılan Problemler

### İleri Yöntem Denemelerinin Sınırlılıkları

Geliştirme sürecinde Dataset Aggregation (DAgger) algoritması, kural tabanlı scripted expert politikası, zamansal model mimarisi ve özel kayıp fonksiyonları gibi gelişmiş teknikler denenmiştir. Bu yaklaşımların her biri, teorik olarak sistem performansını iyileştirme potansiyeline sahiptir.

Kod değişiminden anlaşıldığı kadarıyla, bu tekniklerin pratikte beklenen kazanımı sağlamadığı ya da sistemin yönetilebilirliğini olumsuz etkilediği değerlendirilmiştir. Bu değerlendirme sonucunda çalışma yürütücüsü, önceden stabil olduğu bilinen bir baseline yapıya geri dönme kararı almıştır. Bu süreç, araştırma pratiğinde araştırmacı bütçesi ve zaman kısıtları göz önüne alındığında sıklıkla yaşanan bir yol ayrımını yansıtmaktadır.

### Sim-to-Real Transfer Problemi

Simülasyondan gerçeğe transfer sürecinde en temel zorluk, görsel domain farkının (domain gap) sistematik olarak ele alınmasıdır. Bu sorun yalnızca hedef nokta mimarisi benimsenerek değil, domain randomizasyon uygulanarak ve farklı simülasyon/gerçek veri oranlarını içeren hibrit eğitim stratejileriyle de karşılanmaya çalışılmıştır.

---

## Çözüm Yaklaşımları

Geliştirme süreci boyunca uygulanan temel çözüm stratejileri şöyle özetlenebilir:

1. **Ara Temsil Seçimi:** Doğrudan direksiyon tahmiri yerine hedef nokta tahmiri benimsenerek görsel domain bağımlılığı azaltıldı.
2. **Sistematik Karşılaştırma:** 9 farklı model konfigürasyonu (saf simülasyon, saf gerçek, çeşitli hibrit oranlar) karşılaştırmalı olarak değerlendirildi.
3. **Kontrolcü Kalibrasyonu:** Bias kompansasyonu ve deadband parametreleri eklenerek gerçek araç davranışındaki sistematik sapmalar giderildi.
4. **Sadeliği Tercih Etme:** Karmaşık deneysel bileşenler, güvenilir ve sade bir baseline lehine kaldırıldı.

---

## Sistem Mimarisi

Sistem üç katmanlı bir yapıya sahiptir:

**Algı Katmanı:** Araç ön kamerasından alınan 224×224 piksel görüntü, depthwise separable evrişim katmanlarından oluşan CNN modeline beslenir. Model, ego-frame koordinat sisteminde bir hedef nokta koordinatı `(target_x, target_y)` üretir.

**Karar Katmanı (Kontrolcü):** Geometrik kontrolcü, tahmin edilen koordinattan araç başlığına olan açı hatasını (heading error) hesaplar ve bunu direksiyon komutuna dönüştürür. Gaz komutunu, tahmin edilen hedef noktanın yanal mesafesine göre dinamik olarak ayarlar.

**Dağıtım Katmanı:** Model, gömülü donanımda çalıştırılabilmesi için TensorFlow Lite formatına dönüştürülmüş ve Jetson platformu için optimize edilmiş bir çıkarım modülü geliştirilmiştir.

---

## Test ve Değerlendirme

Sistem performansı iki düzeyde değerlendirilmiştir. Birim testler ile temel bileşenlerin (hedef nokta hesaplama, model değerlendirme) doğruluğu doğrulanmıştır. Kapalı döngü simülasyon değerlendirmeleriyle modelin otonom sürüş kapasitesi ölçülmüştür.

Kapalı döngü değerlendirme, modelin birden fazla simülasyon pistinde otonom olarak çalıştırılmasını ve tamamlanan tur sayısı, pist dışına çıkma sıklığı gibi metriklerin kaydedilmesini kapsamaktadır.

---

## Sonuç

Bu çalışmada, DonkeyCar çerçevesi üzerine inşa edilen bir hedef nokta tabanlı otonom araç kontrol sistemi geliştirilmiştir. Geliştirme süreci; mimari kararlar, sistematik deneyler ve iteratif iyileştirmelerden oluşan tipik bir araştırma döngüsünü yansıtmaktadır.

Hedef nokta yaklaşımı, doğrudan direksiyon tahmiriyle kıyaslandığında sim-to-real transfer için daha uygun bir ara temsil sunmaktadır. Sistematik deneysel çerçeve (9 model konfigürasyonu), farklı veri stratejilerinin etkisinin nesnel olarak değerlendirilmesine olanak tanımıştır.

*Not: Bu paragraflar teze aktarılırken deneysel sonuçlarla desteklenmeli ve gerektiğinde spesifik metrik değerleriyle zenginleştirilmelidir.*
