# Cem Hoca ile Toplantı: Konuşma Metni

Bu dosya, projeyi Cem Hoca'ya baştan sona anlatmak için hazırlandı. Her bölümde önce **ne söyleyeceğin**, ardından gerekirse **ekranda ne göstereceğin** yazıyor. Metni olduğu gibi okuyabilir ya da kendi cümlelerinle anlatabilirsin.

Toplam süre yaklaşık 40 dakika.

| Bölüm | Konu | Süre |
|---|---|---|
| 1 | Projenin amacı | 3 dk |
| 2 | Canlı gösterim | 5 dk |
| 3 | Neden Llama'yı seçtik? | 4 dk |
| 4 | Modelin yapısı: 16 katman | 4 dk |
| 5 | Referans projeyi nasıl inceledik? | 4 dk |
| 6 | Hangi veriyle eğittik? | 5 dk |
| 7 | Modeli nasıl eğittik? Küçük matrisler | 7 dk |
| 8 | Sonuçlar | 6 dk |
| 9 | RAG: modele bilgiyi nasıl verdik? | 3 dk |
| 10 | Sınırlar ve sonraki adımlar | 3 dk |

---

## Toplantıdan önce

1. Bilgisayarı şarja tak.
2. Terminal'de proje klasörüne gir ve uygulamayı eğitilmiş modelle başlat:

   ```bash
   ./start.sh law-qa-r8-best
   ```

3. Tarayıcıda **http://localhost:3000** adresini aç.
4. Arayüzde **Ayarlar**'a tıkla ve sistem mesajına şunu yaz:
   > Sen Türk hukuku konusunda uzman bir asistansın. Soruları Türk mevzuatına ve yargı kararlarına dayanarak kısa, açık ve Türkçe cevapla.
5. Bir deneme sorusu sor. İlk soru biraz yavaş cevaplanır; sonraki sorular hızlanır.
6. VS Code'da şu dosyaları sekmelerde açık tut: `README.md`, `docs/10-law-qa-experiment.md`, `training/configs/law-qa-lora.yaml`.

---

## 1. Projenin amacı (3 dk)

> Hocam, şirketler yapay zekâyı genellikle ChatGPT gibi bulut servisleri üzerinden kullanıyor. Bu durumda yazılan her soru ve yüklenen her doküman başka bir şirketin sunucusuna gidiyor. Sözleşme, müşteri bilgisi ya da kaynak kod gibi gizli veriler için bu büyük bir risk.
>
> Bizim amacımız, bir yapay zekâ modelini tamamen şirketin kendi bilgisayarında çalıştırmak ve şirketin kendi verisiyle eğitebilmek. Kurulumdan sonra internete hiç ihtiyaç duymayan bir sistem kurduk.
>
> Projenin üç parçası var. Birincisi, dili bilen hazır bir model: Llama. İkincisi, bu modelin davranışını şirkete göre uyarlamak için kullandığımız LoRA yöntemi. Üçüncüsü, modele güncel bilgiyi dokümanlardan vermek için kullandığımız RAG yöntemi. Kısaca söylemek gerekirse: **davranışı LoRA ile, bilgiyi RAG ile veriyoruz.**

---

## 2. Canlı gösterim (5 dk)

> Önce sistemi çalışırken göstereyim. Şu an bilgisayarın internet bağlantısını kesiyorum.

**Yap:** Wi-Fi'ı kapat.

> Gördüğünüz sayfa bir internet sitesi değil; tamamen bu bilgisayarın içinde çalışıyor. Adres çubuğunda "localhost" yazıyor, yani "bu bilgisayar" demek.

**Yap:** Bir hukuk sorusu sor, örneğin *Tutuklama sebepleri nelerdir?*

> Cevap internet olmadan geldi. Cevabın altında modelin saniyede kaç kelime parçası ürettiği ve ne kadar bellek kullandığı yazıyor. Model yaklaşık 1 GB bellekle çalışıyor.
>
> Sol üstte "LoRA: law-qa-r8-best" yazıyor. Bu, bizim hukuk verisiyle eğittiğimiz modelin kullanıldığını gösteriyor. Aynı uygulamayı tek bir ayarla eğitilmemiş orijinal modelle de açabiliyoruz.

**Yap:** Wi-Fi'ı tekrar aç.

---

## 3. Neden Llama'yı seçtik? (4 dk)

> Model olarak Meta'nın yayınladığı **Llama 3.2 1B**'yi seçtik. Bu seçimin dört nedeni var.
>
> **Birincisi, Llama bilgisayara indirilip tamamen yerelde çalıştırılabilen bir model.** ChatGPT gibi modellere yalnızca internet üzerinden erişebiliyoruz; Llama'nın ise bütün dosyalarını indirip kendi bilgisayarımızda, internetsiz çalıştırabiliyoruz. Projenin temel şartı da buydu.
>
> **İkincisi, bilgisayarımıza sığıyor.** Elimizde 16 GB bellekli bir MacBook Air var. Referans projede kullanılan Mistral 7B modeli eğitim sırasında 20 GB bellek kullanmış; bizim bilgisayarımızda çalışmazdı. Llama 3.2 1B ise eğitim sırasında yalnızca 2 GB bellek kullandı.
>
> **Üçüncüsü, hızlı.** Bir eğitim yaklaşık 35 dakikada bitiyor. Bu sayede kısa sürede çok sayıda deneme yapabildik.
>
> **Dördüncüsü, lisansı ticari kullanıma izin veriyor.** Bir şirketin bu modeli kendi ürününde kullanmasının önünde bir engel yok.
>
> Modelin 4-bit sıkıştırılmış sürümünü kullanıyoruz. Bu sürümde modelin içindeki sayılar daha az yer kaplıyor; model 2,5 GB yerine 0,7 GB yer tutuyor.
>
> Bu seçimi yapmadan önce yedi modeli karşılaştırdık. Sıradaki aday aynı ailenin biraz daha büyük modeli olan Llama 3.2 3B. Türkçe için de Qwen3 4B'yi değerlendirmeyi düşünüyoruz.

**Göster:** README, Bölüm 4.4'teki model tablosu.

**Hoca Türkçeyi sorarsa:**

> Llama'nın resmî olarak desteklediği diller arasında Türkçe yok. Model Türkçe anlıyor ve yazıyor ama İngilizcedeki kadar iyi değil. Ölçtüğümüzde aynı anlamdaki bir cümle Türkçede İngilizceye göre yaklaşık iki kat fazla parçaya bölünüyor. Bu yüzden eğitimin etkisini çok net görebildik.

---

## 4. Modelin yapısı: 16 katman (4 dk)

> Model, art arda çalışan **16 katmandan** oluşuyor. Bir soru geldiğinde metin sırayla bu 16 katmandan geçiyor; her katman metni bir adım daha işliyor ve en sonda model bir sonraki kelimeyi tahmin ediyor. Cevap, bu tahminin tekrar tekrar yapılmasıyla oluşuyor.
>
> Her katmanın içinde iki bölüm var:
>
> - **Attention:** Kelimelerin birbiriyle ilişkisini kuruyor. Örneğin "kiracı kirayı ödemezse" cümlesinde "ödemezse" kelimesinin "kiracı"ya bağlı olduğunu bu bölüm anlıyor.
> - **MLP:** Attention'dan gelen bilgiyi işleyip bir sonraki katmana aktarıyor. Modelin öğrendiği bilginin büyük kısmı bu bölümde duruyor.
>
> Bu iki bölümün içinde toplam **7 büyük matris** var: attention'da 4, MLP'de 3. Matris dediğimiz şey, sayılardan oluşan büyük bir tablo. Modelin bütün bilgisi bu tablolardaki sayılarda saklı. Modelin toplam 1,24 milyar parametresi var; yani bu tablolarda toplam 1,24 milyar sayı bulunuyor.

**Göster:** Terminal'de şu komutu çalıştır:

```bash
source backend/.venv/bin/activate
```

```bash
python backend/scripts/inspect_model.py
```

> Burada modelin ayar dosyasından okunan bilgiler görünüyor: 16 katman, 2048 boyut, 32 attention head. Altında da katmanların içindeki tabloların adları ve boyutları var.

---

## 5. Referans projeyi nasıl inceledik? (4 dk)

> Önerdiğiniz referans projeyi satır satır inceledik: README dosyasını, notebook'taki komutları ve çıktılarını, veri hazırlama kodlarını.
>
> Eğitmen Mistral 7B modelini, İngiltere'deki bir meslek sınavının (RICS APC) yaklaşık 500 soru-cevabıyla eğitmiş. Modelin 32 katmanının son 16'sına LoRA eklemiş.
>
> İncelerken bazı tutarsızlıklar bulduk:
>
> - Notebook'ta gösterilen komut ile aynı hücrenin çıktısı farklı ayarlara ait. Komutta 200 adım yazıyor, çıktıda 1000 adım çalışmış. Yani yayımlanan sonuç, gösterilen komutla elde edilmemiş.
> - Kaç tur eğitim yapıldığını hesaplarken toplam veri sayısını kullanmış. Oysa eğitimde verinin yalnızca %80'i kullanılıyor. Doğru hesapla sonuç 6,4 değil 8 tur.
> - Eğitimden sonraki test sonucu, eğitimden önceki doğrulama sonucundan daha iyi değil. Bu, modelin ezberlemiş olabileceğine işaret ediyor.
> - Eğitim verisi paylaşılmamış. Bu yüzden aynı veriyle eğitim yapamadık.

**Göster:** `docs/07-reference-project-analysis.md`

---

## 6. Hangi veriyle eğittik? (5 dk)

> Referans projenin verisi paylaşılmadığı için aynı yapıda, ücretsiz ve ticari kullanıma açık bir Türkçe veri seti seçtik: **Türkçe Hukuk Soru-Cevap** veri seti. İçinde Türk hukukuyla ilgili 18 305 soru ve cevabı var. Referans projeyle aynı büyüklükte çalışmak için bunlardan 500 tanesini kullandık.
>
> Yani projemizi bu deneyde **Türk hukuku soru-cevap** alanında özelleştirdik.
>
> Veriyi eğitime vermeden önce temizledik:
>
> - Soruyu cevaplamayıp "daha fazla bilgi için şu yazıya bakın" diyen 194 satırı çıkardık.
> - Tekrar eden 260 soruyu çıkardık.
> - Veriyi 400 eğitim, 49 doğrulama ve 50 test sorusu olarak üçe böldük.
>
> Bölme sırasında önemli bir şey fark ettik. Veride yan yana duran sorular aynı metinden üretilmiş ve birbirine çok benziyor; ölçtüğümüzde rastgele seçilen sorulara göre altı kat daha benzer çıktılar. Soruları rastgele dağıtsaydık, bir test sorusunun neredeyse aynısı eğitim setinde kalırdı ve model cevabı ezberden verirdi. Bunu önlemek için soruları 10'arlı konu grupları halinde böldük. Sonuçta **test soruları, modelin eğitimde hiç görmediği konulardan geliyor.**

**Göster:** `data/law-qa/train.jsonl` dosyasından bir satır. Her satırda bir sistem mesajı, bir soru ve bir cevap var.

---

## 7. Modeli nasıl eğittik? Küçük matrisler (7 dk)

Bu, toplantının en önemli bölümü. Yavaş anlat.

> Bir dil modelini sıfırdan eğitmek binlerce ekran kartı ve aylar gerektiriyor. Biz bunun yerine hazır modeli kendi verimizle bir süre daha eğittik. Buna **fine-tuning** deniyor.
>
> Ama modelin 1,24 milyar sayısının hepsini değiştirmek hem çok bellek ister hem de modelin daha önce öğrendiklerini bozabilir. Bu yüzden **LoRA** yöntemini kullandık.
>
> LoRA'nın fikri şu: Modelin büyük tablolarına hiç dokunmuyoruz, onları donduruyoruz. Her büyük tablonun yanına **iki küçük tablo** ekliyoruz. Model cevap üretirken büyük tablonun sonucuna küçük tabloların sonucunu da ekliyor. Eğitim sırasında yalnızca bu küçük tablolar öğreniyor.
>
> Somut sayılarla anlatayım. Katmanlardaki büyük tablolardan biri 2048'e 2048 boyutunda; yani yaklaşık 4,2 milyon sayı içeriyor. Biz bunun yanına 2048'e 8 ve 8'e 2048 boyutunda iki küçük tablo ekliyoruz. Bu iki tablo toplam 32 768 sayı içeriyor; büyük tablonun yüzde birinden bile az. Buradaki 8 sayısına **rank** deniyor.
>
> Bu küçük tabloları modelin **16 katmanının hepsine**, her katmanda da **7 büyük tablonun hepsinin** yanına ekledik: attention'daki 4 tablo ve MLP'deki 3 tablo. Toplam 16 × 7 = 112 yere ek yaptık. Eğitilen sayıların toplamı 5,6 milyon; bu da modelin yalnızca **%0,46**'sı.
>
> Referans proje 32 katmanın yalnızca son 16'sına ve yalnızca attention tablolarına ek yapmıştı. Biz MLP'yi de dahil ettik, çünkü kullandığımız kütüphanenin güncel sürümünde varsayılan ayar bu.
>
> Eğitimin başında küçük tablolardan biri sıfırlarla başlıyor. Bu yüzden eğitim başladığında model orijinal modelle birebir aynı davranıyor ve oradan yavaş yavaş değişiyor.
>
> Eğitimin sonunda elimize **22 MB'lık bir dosya** geçiyor. Biz buna **adapter** diyoruz. Bu dosyayı modele taktığımızda model eğitilmiş gibi davranıyor, çıkardığımızda orijinal haline dönüyor. Yani her departman için ayrı bir adapter eğitip aynı modele takabiliriz.

**Eğitim ayarlarını anlat:**

> Eğitimde model her adımda 4 soru-cevap gördü. 400 soruyu bir kez baştan sona görmek 100 adım sürüyor; buna bir **epoch** deniyor. Toplam 600 adım, yani 6 epoch eğittik. Referans projenin eğitmeni 5 ile 8 epoch arasını önermişti.
>
> Eğitim sırasında modelin sorunun kendisini değil, yalnızca cevabı öğrenmesini sağladık. Ayrıca her 50 adımda bir modeli, hiç görmediği sorularla ölçüp kaydettik. Böylece modelin en iyi olduğu anı sonradan seçebildik.
>
> Bütün ayarlar, her birinin yanında açıklamasıyla birlikte bu dosyada duruyor.

**Göster:** `training/configs/law-qa-lora.yaml`

**İstersen göster:** Terminal'de eğitim hesaplarını gösteren komutu çalıştır:

```bash
python training/plan.py training/configs/law-qa-lora.yaml
```

---

## 8. Sonuçlar (6 dk)

> Eğitim sırasında modeli iki ayrı şekilde ölçtük: öğrendiği sorulardaki hata ve hiç görmediği sorulardaki hata.

**Göster:** `docs/10-law-qa-experiment.md`, Bölüm 5.1'deki tablo ve grafik.

> Model en iyi haline birinci epoch'un sonunda, yani 100. adımda ulaştı. Sonra öğrendiği sorulardaki hatası azalmaya devam etti ama görmediği sorulardaki hatası artmaya başladı. Bu, modelin ezberlemeye başladığını gösteriyor. Biz de 100. adımdaki modeli seçtik.
>
> Test sonuçları şöyle:
>
> - Eğitilmemiş modelin **perplexity** değeri 27,6. Perplexity, modelin her kelimede ortalama kaç seçenek arasında kaldığını gösteriyor.
> - Eğitimin son adımındaki modelde bu değer 15,1.
> - Bizim seçtiğimiz en iyi anda bu değer **7,1**.
>
> Yani en iyi anı seçmek, son adımı kullanmaya göre sonucu yarı yarıya iyileştirdi. Referans projede bu seçim yapılmıyordu.

**Cevap kalitesini anlat:**

> Aynı 50 test sorusunu dört farklı yöntemle cevaplattık ve cevapları doğru cevaplarla karşılaştırdık. Eğitilmemiş model doğru cevabın beş katı uzunlukta, kendini tekrar eden cevaplar verdi; bazen İngilizce kelimeler karıştırdı. Eğitilmiş model ise kısa ve hukuk diline uygun cevaplar vermeye başladı. Doğru cevaba benzerlik puanı 0,15'ten 0,42'ye çıktı.

**Bir örnek göster:** README Bölüm 6.3'teki "adli para cezası" örneği.

> Burada önemli bir şey görüyoruz: Eğitim, modele cevap vermenin **biçimini** öğretti ama yeni **bilgi** öğretmedi. Eğitilmiş modelin cevabı doğru kalıpta başlıyor, sonra içerik bozuluyor. Çünkü test soruları modelin hiç görmediği konulardan geliyor. Bu, LoRA yönteminin bilinen bir sınırı.

**Sonuçların nerede durduğunu söyle:**

> Bütün sonuçlar proje içinde kayıtlı. Özet tablolar README'de, ayrıntılı deney raporu `docs/10-law-qa-experiment.md` dosyasında. Modelin her test sorusuna verdiği cevaplar ve eğitim kayıtları da `training/results/law-qa` klasöründe duruyor.

---

## 9. RAG: modele bilgiyi nasıl verdik? (3 dk)

> Eğitimin bilgi öğretmediğini gördükten sonra ikinci yöntemi denedik: **RAG**.
>
> RAG'da modeli eğitmiyoruz. Bir soru geldiğinde önce dokümanların içinde o soruyla ilgili metinleri arıyoruz. Bulduğumuz en ilgili üç metni soruyla birlikte modele "kaynak" olarak veriyoruz ve "cevabını yalnızca bu kaynaklara dayandır" diyoruz.
>
> Doküman deposu olarak hukuk veri setindeki yaklaşık 17 800 cevabı kullandık. Test sorularının kendi cevaplarını bu depodan çıkardık; aksi halde model doğru cevabı doğrudan kopyalardı ve ölçüm anlamsız olurdu.
>
> Arama için BM25 adlı, ortak kelimelere bakan klasik bir yöntem kullandık. Ek bir program kurmaya gerek kalmadı.
>
> Sonuç olarak en iyi cevapları **eğitilmiş model ile RAG'ı birlikte** kullandığımızda aldık. Eğitim cevabın biçimini, RAG ise bilgiyi sağlıyor. Az önceki "adli para cezası" sorusuna bu yöntemle doğru cevap geldi: "doğrudan hapis cezasına çevrilebilir."
>
> Arama yöntemimiz henüz basit: cevabı içeren metni soruların yalnızca %14'ünde bulabiliyor. Kelimelere değil anlama bakan bir arama yöntemine geçmek sıradaki işlerimizden biri.

---

## 10. Sınırlar ve sonraki adımlar (3 dk)

> Dürüst olmak gerekirse şu sınırlar var:
>
> - Kullandığımız model küçük; içerik doğruluğu zayıf.
> - Test setimiz 50 sorudan oluşuyor.
> - Kullandığımız benzerlik puanı, cevabın doğru olup olmadığını değil, doğru cevaba ne kadar benzediğini ölçüyor. Doğruluğu bir insan değerlendirmesiyle ölçmemiz gerekiyor.
> - Referans videoyu izleyemedik; yalnızca GitHub'daki dosyaları inceledik.
>
> Şu an farklı ayarlarla karşılaştırmalı eğitimler yapıyoruz: öğrenme hızını, rank değerini ve eğitilen katman sayısını tek tek değiştiriyoruz. Öğrenme hızını yarıya indirdiğimizde model iki kat yavaş öğrendi ama aynı sonuca ulaştı.
>
> Sıradaki adımlarımız şunlar: modeli 1–2 epoch ile yeniden eğitmek, anlama dayalı doküman arama eklemek, aynı deneyi daha büyük Llama 3.2 3B modeliyle tekrarlamak ve sizin vereceğiniz veriyle aynı süreci çalıştırmak.

**Hocaya sor:**

1. Hangi alanda, hangi veriyle çalışmamızı istersiniz?
2. Önceliğimiz eğitim mi olsun, doküman arama mı?
3. Daha büyük bir modele geçelim mi?
4. Başarıyı hangi ölçütle değerlendirmek istersiniz?

---

## Hocanın sorabileceği sorular

**"Neden daha büyük bir model kullanmadınız?"**
> 7 milyar parametreli modeller eğitim sırasında 20 GB'a kadar bellek kullanıyor; bizim bilgisayarımızda 16 GB var. Bizim eğitimimiz 2 GB kullandı, yani daha büyük bir modele geçmek için yerimiz var. Sıradaki aday Llama 3.2 3B.

**"Kaç epoch eğiteceğinize nasıl karar verdiniz?"**
> Kullandığımız kütüphanede doğrudan epoch ayarı yok, adım sayısı veriliyor. Adım sayısını epoch × (eğitim verisi ÷ adım başına örnek) formülüyle hesapladık: 6 × (400 ÷ 4) = 600. Sonra her 50 adımda yaptığımız ölçümlere bakıp en iyi anın 1. epoch olduğunu gördük.

**"Ezberlediğini nasıl anladınız?"**
> Öğrendiği sorulardaki hata 0,29'a kadar indi ama hiç görmediği sorulardaki hata 1,99'dan 2,74'e çıktı.

**"Test sorularını eğitimde görmediğinden emin misiniz?"**
> Evet. Soruları konu gruplarıyla böldük, birbirine çok benzeyen soruları ayrıca kontrol ettik ve birini çıkardık. RAG deposundan da test cevaplarını çıkardık.

**"LoRA modeli bozar mı?"**
> Bozmaz. Modelin orijinal sayılarına dokunmuyoruz; eğitilen kısım ayrı bir dosyada duruyor. Dosyayı çıkarınca model orijinal haline dönüyor.

**"Gerçekten internetsiz çalışıyor mu?"**
> Evet. Wi-Fi kapalıyken denedik. Ayrıca programın internete çıkmasını özel bir ayarla yasaklıyoruz ve API'ye yalnızca bu bilgisayardan erişilebiliyor.

**"Sonuçları tekrar elde edebilir miyiz?"**
> Evet. Aynı ayarlarla eğitimi tekrar çalıştırdık ve birebir aynı sonuçları aldık. Bütün komutlar README'nin 8. bölümünde sırasıyla yazıyor.

**"Kendi verimizle ne kadar sürer?"**
> Veri soru ve cevap sütunlarından oluşan bir tabloya getirilirse aynı araçlar doğrudan çalışıyor. 500 soru-cevaplık bir veriyle eğitim yaklaşık 35 dakika sürüyor.

**"Başka bir bilgisayara kurulabilir mi?"**
> Apple işlemcili herhangi bir Mac'e kurulabilir. Kurulumu tek komutla yapan bir script hazırladık: `./setup.sh`. Adım adım kurulum da README'nin 2. bölümünde anlatılıyor.
