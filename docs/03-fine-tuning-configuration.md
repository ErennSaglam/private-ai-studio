# 03 · Fine-Tuning Configuration

[← README](../README.md)

> Durum (25 Kasım 2026): Parametre isimleri, varsayılan değerler ve davranışlar **kurulu `mlx-lm 0.32.2`** paketinin kaynak kodundan ve `mlx_lm.lora --help` çıktısından doğrulandı. Bu dosyadaki §4 konfigürasyonları genel başlangıç profilleridir (800 örnek varsayımı). Gerçekte çalıştırılan konfigürasyon [training/configs/law-qa-lora.yaml](../training/configs/law-qa-lora.yaml), sonuçları [10-law-qa-experiment.md](10-law-qa-experiment.md) dosyasındadır. Ölçülen eğitim peak memory'si: 2.0 GB (1B, rank 8, batch 4, en uzun örnek 204 token).

Etiketler: **[Kod]** MLX-LM 0.32.2'de doğrulanan davranış · **[Referans]** MoAshour93/MLX_Finetuning · **[Öneri]** bizim başlangıç önerimiz · **[Teori]** literatür.

---

## 1. Referans değerlerin karşılaştırması

Videodan alınan notlar ile referans repodaki iki kaynak **farklı** değerler içeriyor. Video izlenemediği için "video notları" sütunu proje ekibinin notlarıdır.

| Parametre | Video notları | Notebook komut hücresi | Notebook çıktı log'u | README (Step 3.4) | MLX-LM 0.32.2 varsayılanı |
|---|---|---|---|---|---|
| fine_tune_type | lora | lora | — | lora | `lora` |
| num_layers | 16 | 16 | — | 16 | `16` |
| batch_size | 16 | 16 | — (≈1 387 token/adım) | **8** | `4` |
| iters | 200 | 200 | **1000** ("iters: 1000") | **1000** | `1000` |
| val_batches | 4 | 4 | — | **50** | `25` |
| learning_rate | 2e-5 | 2e-5 | **1.000e-05** | **1e-5** | `1e-5` |
| steps_per_report | 10 | 10 | 10 | 10 | `10` |
| steps_per_eval | — | 50 | — | 200 | `200` |
| save_every | — | 50 (yorumda "100") | — | 500 | `100` |
| max_seq_length | — | 2048 | — | 2048 | `2048` |
| grad_checkpoint | — | açık | — | açık | `False` |
| seed | — | 42 | — | 42 | `0` |
| LoRA rank / scale / dropout | — | belirtilmemiş (varsayılan) | — | belirtilmemiş | `8` / `20.0` / `0.0` |

**Sonuç:**
1. Video notları, notebook'taki **komut hücresi** ile birebir aynı.
2. Notebook'taki **çıktı log'u** bu komuta ait değil. Log `iters: 1000` ve `Learning Rate 1.000e-05` gösteriyor. Bu değerler README'deki konfigürasyonla uyuşuyor. Büyük olasılıkla hücre kodu sonradan değiştirilmiş ama yeniden çalıştırılmamış.
3. Dolayısıyla repodaki **ölçülmüş sonuçlar** (train loss eğrisi, test loss 3.945, test ppl 51.684) README konfigürasyonuna (batch 8, 1000 iter, lr 1e-5) aittir. Video notlarındaki konfigürasyona değil.

Ayrıntılı analiz: [07-reference-project-analysis.md](07-reference-project-analysis.md).

## 2. Parametrelerin nasıl verildiği

**[Kod]** İki yol var. YAML dosyası daha tekrarlanabilirdir.

```bash
# CLI
mlx_lm.lora --model models/Llama-3.2-1B-Instruct-4bit --train --data data/hr-qa --batch-size 4 ...

# YAML (önerilen): rank, scale, dropout ve hedef modüller YALNIZCA YAML ile verilebilir
mlx_lm.lora -c configs/lora-balanced.yaml
```

LoRA `rank`, `scale`, `dropout` ve `keys` (hedef modüller) için CLI bayrağı **yoktur**. Bunlar YAML'daki `lora_parameters` bloğunda verilir. CLI'da verilen değerler YAML'daki değerleri geçersiz kılar.

**[Kod]** Eğitim sırasında kullanılan tüm değerler `adapter_path/adapter_config.json` dosyasına yazılır. Her deneyin tam konfigürasyonu böylece kayıt altında kalır.

## 3. Parametreler

Her parametre aynı 10 başlıkla incelenir. "Bizim önerimiz" sütunu Llama 3.2 1B 4-bit ve ~800 eğitim örneği varsayar.

### 3.1 `model`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Base modelin yolu: yerel klasör veya Hugging Face repo adı. |
| 2 | Etkilediği işlem | Dondurulacak $W_{base}$ ağırlıkları, tokenizer ve chat template. |
| 3 | Matematik | $W_{new} = W_{base} + sBA$ ifadesindeki $W_{base}$. |
| 4 | Nasıl belirlenir? | Donanım belleği, dil desteği, lisans ve baseline değerlendirme sonucu. Bkz. [09-model-selection.md](09-model-selection.md). |
| 5 | Büyük model | Daha yüksek kapasite. Bellek ve süre de artar. |
| 6 | Küçük model | Hızlı iterasyon. Kalite tavanı düşüktür. |
| 7 | Kaynak etkisi | Ağırlık belleği: 1B 4-bit 0.70 GB, 3B 4-bit 1.81 GB. Aktivasyon belleği katman sayısı ve $d_{model}$ ile büyür. |
| 8 | Over/underfitting | Büyük model küçük veride daha kolay ezberleyebilir. |
| 9 | Referans | `mistralai/Mistral-7B-Instruct-v0.3`, `mlx_lm.convert` ile **quantize edilmeden** (16-bit) dönüştürülmüş. Log'da 7 248 M parametre görünüyor. |
| 10 | **Bizim önerimiz** | `models/Llama-3.2-1B-Instruct-4bit`. Yerel ve kurulu. 4-bit base olduğu için QLoRA yapılır. Pipeline doğrulandıktan sonra Llama 3.2 3B. |

### 3.2 `data`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | `train.jsonl`, `valid.jsonl` (ve `test.jsonl`) dosyalarını içeren klasör veya bir HF dataset adı. |
| 2 | Etkilediği işlem | Neyin öğrenildiği. |
| 3 | Matematik | Ampirik dağılım $\hat{\mathcal{D}}$; loss bu dağılım üzerinden minimize edilir. |
| 4 | Nasıl belirlenir? | Bkz. [02-dataset-preparation.md](02-dataset-preparation.md). |
| 5–6 | Daha fazla / az veri | Daha fazla temiz veri genellemeyi iyileştirir. Az veri overfitting riskini artırır. |
| 7 | Kaynak etkisi | Süre, toplam token sayısıyla doğrusal artar. Bellek **örnek uzunluğuna** bağlıdır, örnek sayısına bağlı değildir. |
| 8 | Over/underfitting | Az ve tekrarlı veri overfitting'e yol açar. |
| 9 | Referans | RICS APC soru-cevapları, ~500 kayıt, 80/10/10. Validation seti ~50 kayıt. |
| 10 | **Bizim önerimiz** | Yerel bir yol: `data/<dataset-adı>/`. HF dataset adı verilirse internete çıkar; bunu kullanmayacağız. |

### 3.3 `train`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Eğitimi başlatan bayrak. Verilmezse yalnızca `--test` çalışır. |
| 2–8 | | Hiperparametre değildir. Sayısal etkisi yoktur. |
| 9 | Referans | Eğitim hücresinde açık. Test ayrı bir komutla çalıştırılmış. |
| 10 | **Bizim önerimiz** | `train: true`. Testi ayrı çalıştırın; böylece test sonucu seçim sürecine karışmaz. |

### 3.4 `fine_tune_type`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | `lora`, `dora` veya `full`. |
| 2 | Etkilediği işlem | Hangi parametrelerin eğitileceği. |
| 3 | Matematik | **LoRA:** $W + sBA$. **DoRA:** $W' = m \cdot \dfrac{W + sBA}{\lVert W + sBA \rVert_c}$; büyüklük ($m$) ve yön ayrı öğrenilir (Liu et al., 2024). **Full:** tüm $W$ güncellenir. |
| 4 | Nasıl belirlenir? | 16 GB ve 4-bit base için LoRA. |
| 5–6 | | Full fine-tuning: 1.24 B parametre × (ağırlık + gradyan + Adam $m$, $v$) ≈ 1.24 B × 16 bayt ≈ 19.8 GB (fp32, teorik). 16 GB'a sığmaz. |
| 7 | Kaynak etkisi | LoRA'da optimizer durumu yalnızca ~5.6 M parametre içindir. DoRA, LoRA'dan biraz daha pahalıdır. |
| 8 | Over/underfitting | Full fine-tuning daha çok öğrenir, daha çok unutur ("catastrophic forgetting"). |
| 9 | Referans | `lora` |
| 10 | **Bizim önerimiz** | `lora`. DoRA ileride tek değişkenli bir deney olarak denenebilir. |

### 3.5 `num_layers`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Adapter eklenecek transformer bloğu sayısı. **Sondan** sayılır. `-1` tümü demektir. |
| 2 | Etkilediği işlem | Modelin hangi derinliklerinin uyarlanacağı. |
| 3 | Matematik | $P_{LoRA} = \text{num\_layers} \times \sum_{m \in \text{hedefler}} r\,(d_{in}^m + d_{out}^m)$ |
| 4 | Nasıl belirlenir? | Modelin toplam katman sayısına (`num_hidden_layers`) göre. **Rank ile ve toplam katman sayısıyla karıştırılmamalıdır.** |
| 5 | Artırılırsa | Eğitilebilir parametre doğrusal artar. Backward'da gradyan hesaplanan katman sayısı artar, bu yüzden süre ve bellek de artar. |
| 6 | Azaltılırsa | Yalnızca üst katmanlar uyarlanır. Daha hızlıdır, daha sınırlıdır. |
| 7 | Kaynak etkisi | 1B, r=8: 4 katman 1.41 M, 8 katman 2.82 M, 16 katman 5.64 M parametre **[Ölçüm]**. |
| 8 | Over/underfitting | Fazla katman + küçük veri, overfitting riskini artırır. |
| 9 | Referans | 16 / 32 (Mistral). Eğitmenin gerekçesi: "toplam katmanların %40–80'i". Bu **eğitmenin kendi önerisidir**, MLX dokümantasyonunda bu oran geçmez. |
| 10 | **Bizim önerimiz** | 1B modelde `16`. Bu değer 1B'nin **tüm** katmanlarıdır (toplam 16). Bellek darsa `8`. |

### 3.6 `batch_size`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Bir adımda işlenen örnek sayısı. |
| 2 | Etkilediği işlem | Gradyan tahmininin gürültüsü ve adım başına bellek. |
| 3 | Matematik | **[Kod]** $\mathcal{L}_{batch} = \dfrac{\sum_{b,i} m_{b,i}\,\ell_{b,i}}{\sum_{b,i} m_{b,i}}$. Örnek başına değil, **maskelenmiş token başına** ortalama alınır. |
| 4 | Nasıl belirlenir? | Belleğe sığan en büyük değer. Daha büyük etkin batch için `grad_accumulation_steps` kullanılır. |
| 5 | Artırılırsa | Gradyan daha az gürültülü olur. Aktivasyon belleği yaklaşık doğrusal artar. Aynı `iters` ile daha çok epoch yapılır. |
| 6 | Azaltılırsa | Bellek azalır, gradyan gürültüsü artar. Gürültü bazen genellemeye yardım eder. |
| 7 | Kaynak etkisi | Aktivasyon belleği ∝ $B \times T$. Logit tensörü $B \times T \times 128\,256$ boyutundadır ve büyük vocab nedeniyle baskındır. Bkz. [04 §9](04-training-calculations.md). |
| 8 | Over/underfitting | Sabit `iters` ile büyük batch daha çok epoch yapar ve overfitting'e itebilir. |
| 9 | Referans | Komut hücresi 16 ("gradyan kararlılığı için"), README 8. |
| 10 | **Bizim önerimiz** | `4` (MLX-LM varsayılanı). **[Kod]** Batch'ler uzunluğa göre sıralanmış örneklerden oluşturulur ve $N \bmod B$ adet **en uzun** örnek hiç kullanılmaz. $N$ değeri $B$'nin katı seçilmelidir. |

### 3.7 `iters`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Toplam eğitim adımı (batch) sayısı. MLX-LM'de `epochs` parametresi **yoktur**. |
| 2 | Etkilediği işlem | Toplam öğrenme miktarı. |
| 3 | Matematik | **[Kod]** $\text{epoch} = \dfrac{\text{iters}}{\lfloor N/B \rfloor}$ |
| 4 | Nasıl belirlenir? | Hedef epoch sayısından geriye hesaplanır. Sonra validation eğrisine göre ayarlanır. |
| 5 | Artırılırsa | Daha çok öğrenme. Belli bir noktadan sonra overfitting. |
| 6 | Azaltılırsa | Underfitting riski. |
| 7 | Kaynak etkisi | Süre doğrusal artar. Belleği etkilemez. |
| 8 | Over/underfitting | Doğrudan etkiler. En önemli kontrol düğmelerinden biridir. |
| 9 | Referans | Hücrede 200, README ve log'da 1000. Eğitmen 5–8 epoch hedeflemiş ve $200 \times 16 / 500 = 6.4$ epoch hesaplamış. Ancak 500 **toplam** kayıt sayısıdır; train seti ~400 kayıttır. Doğru hesap: $\lfloor 400/16 \rfloor = 25$ adım/epoch, $200/25 = 8$ epoch. README konfigürasyonunda ise $1000 / \lfloor 400/8 \rfloor = 20$ epoch olur. |
| 10 | **Bizim önerimiz** | 2–3 epoch ile başla. 800 örnek ve B=4 için $\lfloor 800/4 \rfloor = 200$ adım/epoch, 3 epoch = **600 iters**. |

### 3.8 `val_batches`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Her validation ölçümünde kullanılan batch sayısı. `-1` tüm validation setini kullanır. |
| 2 | Etkilediği işlem | Validation loss'un ne kadar örnekle hesaplandığı. |
| 3 | Matematik | **[Kod]** Kullanılan batch sayısı $= \min(\lfloor N_{val}/B \rfloor, \text{val\_batches})$. Kullanılan örnek sayısı bunun $B$ katıdır. |
| 4 | Nasıl belirlenir? | Küçük validation setlerinde `-1`. |
| 5 | Artırılırsa | Daha güvenilir ölçüm. Validation süresi uzar. |
| 6 | Azaltılırsa | **[Kod]** Her validation çağrısında batch'ler yeniden karıştırılır. Tüm set kullanılmazsa her ölçüm **farklı bir alt kümede** yapılır ve eğri gürültülü olur. |
| 7 | Kaynak etkisi | Yalnızca forward pass olduğu için bellek etkisi küçüktür. Süreye etkisi vardır. |
| 8 | Over/underfitting | Doğrudan etkisi yok. Ancak gürültülü ölçüm, yanlış checkpoint seçimine yol açar. |
| 9 | Referans | Hücrede 4. Eğitmenin formülü: $\lceil N_{val}/B \rceil$. Ama yorumda "Ceiling(500/16) = 4" yazıyor; bu hesap 32 eder. Doğru girdi 50 validation kaydıdır: $\lceil 50/16 \rceil = 4$. **[Kod]** MLX-LM aşağı yuvarlar ($\lfloor 50/16 \rfloor = 3$), yani 4 istense de 3 batch, 48 örnek kullanılır. |
| 10 | **Bizim önerimiz** | `-1` |

### 3.9 `learning_rate`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Optimizer adım büyüklüğü $\eta$. |
| 2 | Etkilediği işlem | Adapter parametrelerinin değişim hızı. |
| 3 | Matematik | Adam'da her adımda parametre değişimi yaklaşık $\eta$ kadardır. Bkz. [01 §12](01-mathematical-foundations.md#12-adam-optimizer). Çıktıdaki etki $s \cdot \Delta(BA)$ olduğu için `scale` ile **çarpımsal** olarak etkileşir. |
| 4 | Nasıl belirlenir? | Varsayılandan başla, sonra $\{0.5\times, 1\times, 2\times\}$ taraması yap. |
| 5 | Artırılırsa | Hızlı öğrenme. Kararsızlık, loss sıçramaları, base yeteneklerin unutulması. |
| 6 | Azaltılırsa | Yavaş öğrenme, underfitting. |
| 7 | Kaynak etkisi | Bellek ve adım süresi değişmez. Yakınsama süresi değişir. |
| 8 | Over/underfitting | Yüksek LR hızla overfit eder. Düşük LR underfit eder. |
| 9 | Referans | Hücrede 2e-5, README ve log'da 1e-5. Eğitmenin görüşü: "normal aralık 2e-5 – 4e-5". Bu **eğitmenin görüşüdür**. |
| 10 | **Bizim önerimiz** | `1e-5` (MLX-LM varsayılanı, `scale = 20` ile birlikte). PEFT tabanlı rehberlerdeki 1e-4 – 2e-4 değerleri $s = \alpha/r \approx 2$ içindir; MLX'e doğrudan taşınmamalıdır. İsteğe bağlı: kısa warmup ve cosine decay (aşağıdaki YAML örneklerine bakın). |

### 3.10 `steps_per_report`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Kaç adımda bir train loss yazdırılacağı. |
| 2 | Etkilediği işlem | Yalnızca loglama. |
| 3 | Matematik | **[Kod]** Raporlanan train loss, son `steps_per_report` adımın ortalamasıdır. |
| 4–8 | | Eğitime etkisi yoktur. Çok küçük değer log'u gürültülü yapar. |
| 9 | Referans | 10 |
| 10 | **Bizim önerimiz** | `10` |

### 3.11 `steps_per_eval`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Kaç adımda bir validation loss ölçüleceği. |
| 2 | Etkilediği işlem | İzleme sıklığı ve checkpoint seçimi çözünürlüğü. |
| 3 | Matematik | **[Kod]** Validation 1. adımda, her `steps_per_eval` katında ve son adımda çalışır. Ölçüm sayısı ≈ $1 + \lfloor \text{iters}/\text{steps\_per\_eval} \rfloor$. |
| 4 | Nasıl belirlenir? | Eğitim boyunca 8–12 ölçüm alınacak şekilde. |
| 5 | Artırılırsa | Daha seyrek ölçüm, daha az ek süre. |
| 6 | Azaltılırsa | Daha sık ölçüm. Toplam süre uzar. |
| 7 | Kaynak etkisi | Her ölçüm $N_{val}$ örnek üzerinde bir forward pass yapar. |
| 8 | Over/underfitting | Overfitting'in **tespit edilebilmesini** sağlar. |
| 9 | Referans | Hücrede 50, README'de 200. Eğitmenin formülü: $\max(50, \text{iters}/\!/10)$. Ancak yorumda iters=200 yerine 250 kullanılmış. Sonuç yine 50. |
| 10 | **Bizim önerimiz** | `50` (600 iters için 13 ölçüm). |

### 3.12 `adapter_path`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Adapter ağırlıklarının ve konfigürasyonun kaydedileceği klasör. |
| 2 | Etkilediği işlem | **[Kod]** Bu klasöre `adapters.safetensors` (son durum), `NNNNNNN_adapters.safetensors` (checkpoint'ler) ve `adapter_config.json` yazılır. |
| 3 | Matematik | Saklanan değerler: tüm $A_m$, $B_m$ matrisleri. |
| 4–8 | | Sayısal etkisi yoktur. |
| 9 | Referans | `${adapters}` değişkeni. |
| 10 | **Bizim önerimiz** | Deney başına ayrı klasör: `adapters/<tarih>-<deney-id>/`. Asla üzerine yazmayın. Backend entegrasyonu: `config.ADAPTER_PATH`. |

### 3.13 `save_every`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Kaç adımda bir checkpoint kaydedileceği. |
| 2 | Etkilediği işlem | Kesintiye karşı koruma ve "en iyi checkpoint" seçimi. |
| 3 | Matematik | Checkpoint sayısı $= \lfloor \text{iters}/\text{save\_every} \rfloor$ (+ sonda `adapters.safetensors`). |
| 4 | Nasıl belirlenir? | `steps_per_eval` ile **eşit** olmalı. Böylece her validation ölçümünün bir checkpoint karşılığı olur. |
| 5–6 | | Sık kayıt disk kullanır. Seyrek kayıt en iyi noktayı kaçırabilir. |
| 7 | Kaynak etkisi | 1B, r=8: 5.64 M parametre × 4 bayt ≈ 22.5 MB/checkpoint (float32 varsayımıyla, teorik). |
| 8 | Over/underfitting | Validation minimumundaki checkpoint'i seçmek, manuel **early stopping** yapmaktır. |
| 9 | Referans | Hücrede 50, yorumda "100", README'de 500. Eğitmenin gerekçesi: "en az 4 checkpoint olmalı". |
| 10 | **Bizim önerimiz** | `50` (= `steps_per_eval`). |

### 3.14 `max_seq_length`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Bir örnekte işlenecek en fazla token. Uzun örnekler **kesilir**. |
| 2 | Etkilediği işlem | Bellek, süre ve kesilen içerik. |
| 3 | Matematik | Attention hesabı $O(T^2)$, diğer aktivasyonlar $O(T)$. **[Kod]** Batch, en uzun örneğe göre 32'nin katına yuvarlanarak pad edilir: $T_{pad} = 1 + 32\lceil T_{max}/32 \rceil$. |
| 4 | Nasıl belirlenir? | Token istatistiklerine göre. Bkz. [02 §6](02-dataset-preparation.md#6-token-uzunlukları). |
| 5 | Artırılırsa | Uzun örnek yoksa etkisi **yoktur**; padding gerçek uzunluğa göre yapılır. Uzun örnek varsa bellek artar. |
| 6 | Azaltılırsa | Bellek tavanı düşer, ama cevaplar kesilebilir. |
| 7 | Kaynak etkisi | Peak memory en uzun batch tarafından belirlenir. |
| 8 | Over/underfitting | Kesilen cevaplar modele "yarım cevap" öğretebilir. |
| 9 | Referans | 2048. Gerekçe: her kayıt ≤ 200 token. |
| 10 | **Bizim önerimiz** | Ölçülen max uzunluğun üstünde en küçük 2'nin kuvveti. Bilinmiyorsa `2048`. |

### 3.15 `grad_checkpoint`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Gradient checkpointing. Aktivasyonlar saklanmaz, backward sırasında yeniden hesaplanır. |
| 2 | Etkilediği işlem | Bellek ve süre dengesi. |
| 3 | Matematik | **[Teori]** Katman başına yalnızca girdi saklanır: $B \cdot T \cdot d$. Fazladan bir forward pass yapılır (Chen et al., 2016). |
| 4 | Nasıl belirlenir? | Bellek darsa açılır. |
| 5–6 | Açık / kapalı | Açık: bellek belirgin azalır, adım süresi artar (teorik olarak ~%30 civarı; ölçülmedi). Kapalı: tersi. |
| 7 | Kaynak etkisi | 16 GB'ta 3B ve üstü modeller için büyük olasılıkla gerekli. |
| 8 | Over/underfitting | Etkisi yok. Matematiksel sonuç aynıdır. |
| 9 | Referans | Açık. |
| 10 | **Bizim önerimiz** | `true`. 1B ile bellek bolsa kapatılıp hız farkı **ölçülebilir**. |

### 3.16 `grad_accumulation_steps`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Optimizer güncellemesinden önce kaç batch'in gradyanının biriktirileceği. |
| 2 | Etkilediği işlem | Etkin batch boyutu. |
| 3 | Matematik | $B_{eff} = B \times k$. **[Kod]** Gradyanlar toplanır, $k$'ye bölünür ve `it % k == 0` adımlarında güncelleme yapılır. Optimizer güncelleme sayısı $= \text{iters}/k$. |
| 4 | Nasıl belirlenir? | Hedef etkin batch, belleğe sığmıyorsa. |
| 5 | Artırılırsa | Daha az gürültü, daha az güncelleme. Aynı `iters` ile daha az öğrenme. |
| 6 | Azaltılırsa | Tersi. |
| 7 | Kaynak etkisi | Bellek $B$'ye göre belirlenir, $k$ küçük bir ek maliyet getirir (biriken gradyan, LoRA'da ~22 MB). |
| 8 | Over/underfitting | Dolaylı etki. |
| 9 | Referans | **Kullanılmamış.** Referans notebook'taki eski sürümün help çıktısında bu bayrak yok. |
| 10 | **Bizim önerimiz** | `1`. Minimum bellek konfigürasyonunda `2`. |

### 3.17 `seed`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Rastgele sayı üreteci tohumu. |
| 2 | Etkilediği işlem | **[Kod]** `mx.random.seed` (LoRA $A$ başlatma, dropout) ve `np.random.seed` (batch sırası). |
| 3 | Matematik | — |
| 4 | Nasıl belirlenir? | Sabit bir değer. Önemli sonuçlar için 3 farklı seed ile tekrar edilir. |
| 5–8 | | Sonuç varyansı ölçülmeden iki deney arasındaki küçük fark anlamlı kabul edilmemelidir. |
| 9 | Referans | 42 |
| 10 | **Bizim önerimiz** | `42`. Kritik karşılaştırmalarda `{42, 43, 44}`. |

### 3.18 LoRA `rank`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | $A$ ve $B$ matrislerinin iç boyutu $r$. |
| 2 | Etkilediği işlem | Adapter kapasitesi. |
| 3 | Matematik | Modül başına $r(d_{in} + d_{out})$ parametre. $\text{rank}(\Delta W) \le r$. |
| 4 | Nasıl belirlenir? | Göreve göre. Stil ve format için düşük, alan uyarlaması için daha yüksek. |
| 5 | Artırılırsa | Parametre sayısı doğrusal artar. Overfitting riski artar. |
| 6 | Azaltılırsa | Kapasite azalır, underfitting olabilir. |
| 7 | Kaynak etkisi | 1B: r=8 → 5.64 M, r=16 → 11.27 M **[Ölçüm]**. Aktivasyon belleğine etkisi küçüktür. |
| 8 | Over/underfitting | Doğrudan etkiler. |
| 9 | Referans | Belirtilmemiş, varsayılan kullanılmış. Log'da 3.408 M eğitilebilir parametre var. Bu sayı Mistral 7B için iki şekilde elde edilebilir: (a) r=8 ile q, k, v, o projeksiyonları, ya da (b) r=16 ile yalnızca q ve v. Hangisi olduğu repodan anlaşılamıyor. |
| 10 | **Bizim önerimiz** | `8`. Deneylerde $\{4, 8, 16, 32\}$. |

### 3.19 LoRA `scale` (alpha karşılığı)

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | $\Delta W$ çarpanı $s$. MLX-LM'de **`scale`** adını taşır. **`alpha` parametresi yoktur.** |
| 2 | Etkilediği işlem | Adapter'ın çıktıya etkisinin büyüklüğü. |
| 3 | Matematik | MLX: $W + s\,BA$. PEFT: $W + \frac{\alpha}{r}BA$. Eşdeğer $\alpha = s \cdot r$. |
| 4 | Nasıl belirlenir? | Learning rate ile birlikte. İkisini aynı anda değiştirmeyin. |
| 5 | Artırılırsa | Etkin adım büyür (yüksek LR'ye benzer). |
| 6 | Azaltılırsa | Etkin adım küçülür. |
| 7 | Kaynak etkisi | Yok. |
| 8 | Over/underfitting | Yüksek scale ve yüksek LR birlikte kararsızlık riski taşır. |
| 9 | Referans | Belirtilmemiş. |
| 10 | **Bizim önerimiz** | `20.0` (varsayılan). **[Kod]** YAML'a `alpha` yazılırsa 0.32.2 `KeyError` verir. |

### 3.20 LoRA `dropout`

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Adapter girdisine uygulanan dropout olasılığı $p$. |
| 2 | Etkilediği işlem | **[Kod]** `z = (dropout(x) @ A) @ B`. Yalnızca adapter yolunu etkiler, base yolu etkilemez. |
| 3 | Matematik | Eğitimde $x_j$ değeri $p$ olasılıkla 0 yapılır, kalanlar $1/(1-p)$ ile ölçeklenir. |
| 4 | Nasıl belirlenir? | Overfitting görülürse açılır. |
| 5–6 | | Yüksek dropout daha fazla regularization ve daha yavaş öğrenme demektir. |
| 7 | Kaynak etkisi | İhmal edilebilir. |
| 8 | Over/underfitting | Overfitting'e karşı bir araçtır. |
| 9 | Referans | Belirtilmemiş (varsayılan 0.0). |
| 10 | **Bizim önerimiz** | `0.0`. Küçük veride overfitting varsa `0.05`–`0.1`. |

### 3.21 Target modules (`lora_parameters.keys`)

| # | Başlık | Açıklama |
|---|---|---|
| 1 | Nedir? | Adapter eklenecek lineer katmanlar. |
| 2 | Etkilediği işlem | Modelin hangi alt sistemlerinin uyarlanacağı. |
| 3 | Matematik | $P = \text{num\_layers} \sum_{m \in \text{keys}} r(d_{in}^m + d_{out}^m)$ |
| 4 | Nasıl belirlenir? | **[Kod]** 0.32.2'de `keys` verilmezse, bloktaki **tüm** lineer katmanlar hedeflenir: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (1B modelde doğrulandı **[Ölçüm]**). |
| 5 | Daha fazla modül | Daha yüksek kapasite. MLP modülleri büyüktür: 1B'de r=8 için blok başına 245 760 parametre, attention için 106 496. |
| 6 | Daha az modül | Örneğin yalnızca `q_proj` ve `v_proj` (orijinal LoRA makalesindeki seçim): blok başına 53 248 parametre. |
| 7 | Kaynak etkisi | Parametre ve hesap sayısı modüllerle orantılıdır. |
| 8 | Over/underfitting | Kapasite ile ilişkilidir. |
| 9 | Referans | Belirtilmemiş. Eski sürümün varsayılanı kullanılmış (bkz. 3.18). |
| 10 | **Bizim önerimiz** | Varsayılan (tüm lineer katmanlar). Deney: yalnızca attention modülleri ve tümü karşılaştırması. |

### 3.22 Ek: `mask_prompt`, `optimizer`, `lr_schedule`, `test_batches`

| Parametre | MLX-LM varsayılanı | Öneri | Not |
|---|---|---|---|
| `mask_prompt` | `false` | **`true`** | Loss yalnızca assistant cevabından hesaplanır. Bkz. [02 §10](02-dataset-preparation.md#10-assistant-only-loss-masking). |
| `optimizer` | `adam` | `adam` | `adamw` ile weight decay ayrı bir deney olabilir. |
| `lr_schedule` | yok (sabit) | İsteğe bağlı | `{name, arguments, warmup, warmup_init}`. `name`, `mlx.optimizers.schedulers` içinden bir fonksiyondur (örneğin `cosine_decay`). |
| `test_batches` | `500` | `-1` | Test setinin tamamı. |

---

## 4. Donanıma özel eğitim stratejisi (MacBook Air M3, 16 GB)

**Ölçülen sınır:** `mx.device_info()` → `max_recommended_working_set_size = 12.71 GB`, `memory_size = 17.18 GB` **[Ölçüm]**. macOS ve diğer uygulamalar da bellek kullanır. Eğitim sırasında tarayıcı ve IDE kapalı olmalıdır.

**MacBook Air'e özgü not:** Air modelinde fan yoktur. Uzun eğitimlerde termal kısıtlama (throttling) nedeniyle adım süresi zamanla uzayabilir. Süre ölçümleri bu yüzden eğitim boyunca izlenmelidir (`It/sec` sütunu).

> ⚠️ Aşağıdaki üç konfigürasyon **kesin optimum değildir**. Bunlar ölçülmemiş, deneysel **başlangıç** konfigürasyonlarıdır. Bellek etkileri teorik değerlendirmedir. İlk deneylerde `Peak mem` değeri log'dan okunmalı ve bu dosya güncellenmelidir.

Varsayım: Llama 3.2 1B Instruct 4-bit, `data/hr-qa/` içinde 800 train, 100 valid, 100 test örneği. Örnek başına ortalama ~300, en fazla ~900 token.

### 4.1 Minimum bellek kullanımı

```yaml
# configs/lora-min-memory.yaml  (planlandı — henüz oluşturulmadı)
# Amaç: en düşük peak memory. Pipeline'ı doğrulamak ve 16 GB'ta güvenli çalışmak.
model: models/Llama-3.2-1B-Instruct-4bit   # 4-bit base → QLoRA
train: true
data: data/hr-qa                           # train/valid/test.jsonl
fine_tune_type: lora
mask_prompt: true                          # loss yalnızca assistant cevabından
num_layers: 8                              # 16 katmandan yalnızca SON 8'i (blok 8–15)
batch_size: 2                              # küçük batch → aktivasyon belleği yarıya iner (B=4'e göre)
grad_accumulation_steps: 2                 # etkin batch = 2 × 2 = 4
iters: 1200                                # floor(800/2) = 400 adım/epoch → 3 epoch; 600 optimizer güncellemesi
val_batches: -1                            # tüm validation seti → tutarlı ölçüm
learning_rate: 1.0e-5                      # MLX-LM varsayılanı (scale=20 ile birlikte)
steps_per_report: 20
steps_per_eval: 100                        # 13 validation ölçümü
save_every: 100                            # her validation'a karşılık bir checkpoint
adapter_path: adapters/2026xxxx-min-memory
max_seq_length: 1024                       # veri istatistiğine göre; max ~900 varsayımı
grad_checkpoint: true                      # aktivasyonları sakla değil, yeniden hesapla
seed: 42
lora_parameters:
  rank: 4                                  # düşük kapasite: stil/format uyarlaması için
  scale: 20.0                              # MLX'te alpha yok; ΔW = scale · BA
  dropout: 0.0
# Eğitilebilir parametre: 8 katman × 4 × 44 032 = 1 409 024 (rank 8 / 4 katman ile aynı sayı)
```

Eğitilebilir parametre sayısı: $8 \times 4 \times 44\,032 = 1\,409\,024$. Burada 44 032, blok başına $\sum (d_{in}+d_{out})$ değeridir: $352\,256 / 8$.

### 4.2 Dengeli eğitim (ilk ana deney)

```yaml
# configs/lora-balanced.yaml  (planlandı)
# Amaç: makul kalite / süre dengesi. Deney serisinin "kontrol" konfigürasyonu.
model: models/Llama-3.2-1B-Instruct-4bit
train: true
data: data/hr-qa
fine_tune_type: lora
mask_prompt: true
num_layers: 16                             # 1B modelin TÜM 16 katmanı (-1 ile aynı)
batch_size: 4                              # MLX-LM varsayılanı; 800 = 4'ün katı → hiç örnek atılmaz
grad_accumulation_steps: 1
iters: 600                                 # floor(800/4) = 200 adım/epoch → 3 epoch
val_batches: -1
learning_rate: 1.0e-5
steps_per_report: 10
steps_per_eval: 50                         # 13 ölçüm (1, 50, 100, …, 600)
save_every: 50
adapter_path: adapters/2026xxxx-balanced
max_seq_length: 2048
grad_checkpoint: true
seed: 42
lora_parameters:
  rank: 8                                  # MLX-LM varsayılanı
  scale: 20.0
  dropout: 0.0
  # keys verilmedi → tüm lineer katmanlar (q,k,v,o,gate,up,down)
# Eğitilebilir parametre: 5 636 096 (%0.456) — ölçüldü
```

### 4.3 Daha yüksek adaptör kapasitesi

```yaml
# configs/lora-high-capacity.yaml  (planlandı)
# Amaç: daha fazla bilgi/alan uyarlaması. Overfitting riski en yüksek konfigürasyon.
model: models/Llama-3.2-1B-Instruct-4bit
train: true
data: data/hr-qa
fine_tune_type: lora
mask_prompt: true
num_layers: -1                             # tüm katmanlar (1B için = 16)
batch_size: 4
grad_accumulation_steps: 2                 # etkin batch 8 → daha kararlı gradyan
iters: 600                                 # 3 epoch; 300 optimizer güncellemesi
val_batches: -1
learning_rate: 1.0e-5
lr_schedule:                               # kısa warmup + cosine decay
  name: cosine_decay
  warmup: 30                               # ilk 30 adımda 1e-7 → 1e-5 doğrusal artış
  warmup_init: 1.0e-7
  arguments: [1.0e-5, 600, 1.0e-6]         # (başlangıç LR, decay adımı, son LR)
steps_per_report: 10
steps_per_eval: 50
save_every: 50
adapter_path: adapters/2026xxxx-high-capacity
max_seq_length: 2048
grad_checkpoint: true
seed: 42
lora_parameters:
  rank: 16                                 # 2× kapasite
  scale: 20.0
  dropout: 0.05                            # yüksek rank'a karşı hafif regularization
# Eğitilebilir parametre: 11 272 192 (%0.91) — ölçüldü
```

### 4.4 Karşılaştırma

| | Minimum bellek | Dengeli | Yüksek kapasite |
|---|---|---|---|
| num_layers / toplam | 8 / 16 | 16 / 16 | 16 / 16 |
| rank | 4 | 8 | 16 |
| batch × accum = etkin | 2 × 2 = 4 | 4 × 1 = 4 | 4 × 2 = 8 |
| iters / optimizer adımı | 1200 / 600 | 600 / 600 | 600 / 300 |
| epoch | 3 | 3 | 3 |
| max_seq_length | 1024 | 2048 | 2048 |
| Eğitilebilir parametre | 1.41 M | 5.64 M | 11.27 M |
| Göreli bellek (teorik) | en düşük | orta | orta (B aynı) |
| Peak memory (ölçüm) | — | — | — |
| Süre (ölçüm) | — | — | — |

Not: Bu üç konfigürasyon **birden çok değişkeni birlikte** değiştirir. Bunlar başlangıç profilleridir, kontrollü deney değildir. Kontrollü deneylerde her seferinde tek bir değişken değişir. Bkz. [06-experimental-design.md](06-experimental-design.md).

### 4.5 3B modele geçiş (sonraki adım)

Llama 3.2 3B 4-bit için dengeli konfigürasyon şöyle uyarlanabilir: `num_layers: 16` (toplam 28'in son 16'sı), r=8 için 6.95 M eğitilebilir parametre (hesaplandı). Önce `batch_size: 2` ile peak memory ölçülmeli, sonra 4'e çıkılmalıdır.

---

## Kaynaklar

- MLX-LM 0.32.2: `mlx_lm/lora.py` (`CONFIG_DEFAULTS`, argparse), `mlx_lm/tuner/trainer.py`, `mlx_lm/tuner/utils.py`, `mlx_lm/tuner/lora.py`
- MLX-LM LoRA rehberi: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- Hu et al., *LoRA*, 2021. arXiv:2106.09685
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, 2024. arXiv:2402.09353
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016. arXiv:1604.06174
- Referans: MoAshour93/MLX_Finetuning, `Mistral_Instruct_v03_MLX_Finetuning-Github.ipynb` hücre 13
