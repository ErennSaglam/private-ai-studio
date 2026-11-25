# 04 · Training Calculations

[← README](../README.md)

> Bu dosyadaki hesapların hepsi **teoriktir**. Ölçülmüş değerler **[Ölçüm]** (bu proje) veya **[Referans]** (MoAshour93 notebook'u) etiketiyle ayrıca gösterilir. Eğitim süreleri ölçülmedi. Süre formülleri, ölçülecek bir throughput değeri için şablondur.

Model: Llama 3.2 1B Instruct 4-bit ($L = 16$, $d = 2048$, $d_{ff} = 8192$, $V = 128\,256$).

---

## 1. Steps per epoch

**[Kod]** MLX-LM örnekleri uzunluğa göre sıralar, ardışık $B$ örnekten sabit batch'ler oluşturur ve her geçişte **batch sırasını** karıştırır. Artan örnekler bir batch oluşturmaz.

$$S_{epoch} = \left\lfloor \frac{N_{train}}{B} \right\rfloor, \qquad N_{dışarıda} = N_{train} \bmod B$$

Dışarıda kalan $N \bmod B$ örnek, sıralamanın sonundaki **en uzun** örneklerdir ve eğitimde **hiç** görülmezler. $N$ değerini $B$'nin katı seçin.

## 2. Total iterations ve yaklaşık epoch

$$\text{epoch} = \frac{\text{iters}}{S_{epoch}} \qquad\Longleftrightarrow\qquad \text{iters} = \text{epoch} \times S_{epoch}$$

`iters` değeri $S_{epoch}$'un tam katı değilse son epoch **kısmidir**. Batch sırası her epoch'ta yeniden karıştırıldığı için kısmi epoch rastgele bir batch alt kümesi görür.

## 3. Effective batch size ve optimizer adımı

$$B_{eff} = B \times k_{accum}, \qquad U = \left\lfloor \frac{\text{iters}}{k_{accum}} \right\rfloor \quad (\text{optimizer güncelleme sayısı})$$

## 4. Total processed samples ve tokens

$$N_{işlenen} = \text{iters} \times B$$

$$T_{işlenen} \approx \text{iters} \times B \times \bar{\ell}$$

Burada $\bar{\ell}$ chat template dahil ortalama örnek uzunluğudur.

Hesaba girmesi gereken üç düzeltme:

1. **Padding:** Batch, en uzun örneğe göre pad edilir: $T_{pad} = 1 + 32\lceil \ell_{max}/32 \rceil$. Batch'ler uzunluğa göre sıralı örneklerden oluştuğu için padding israfı küçüktür, ama sıfır değildir. Hesaplama maliyeti $B \times T_{pad}$ ile orantılıdır.
2. **Loss'a giren token:** `mask_prompt: true` ile yalnızca cevap tokenları sayılır. MLX-LM log'undaki "Trained Tokens" sütunu maske toplamını gösterir.
3. **Değişken uzunluk:** Her batch'in maliyeti farklıdır. Adım süresi sabit değildir. Referans log'unda `It/sec` 0.155 ile 0.249 arasında değişiyor **[Referans]**.

## 5. Validation batch hesapları

$$V_{batch} = \min\!\left(\left\lfloor \frac{N_{val}}{B}\right\rfloor,\ \text{val\_batches}\right) \quad (\text{val\_batches} = -1 \Rightarrow \lfloor N_{val}/B \rfloor)$$

$$\text{ölçüm sayısı} = 1 + \left\lfloor\frac{\text{iters}}{\text{steps\_per\_eval}}\right\rfloor \ (+1 \text{ eğer iters, steps\_per\_eval'in katı değilse})$$

**[Referans] örneği:** $N_{val} = 50$, $B = 16$, `val_batches = 4`:

$$V_{batch} = \min(\lfloor 50/16 \rfloor, 4) = \min(3, 4) = 3 \text{ batch} = 48 \text{ örnek}$$

Eğitmen $\lceil 50/16 \rceil = 4$ hesaplamış. MLX-LM ise aşağı yuvarlar.

## 6. Checkpoint sayısı

$$C = \left\lfloor \frac{\text{iters}}{\text{save\_every}} \right\rfloor \quad (+\ \text{sonda } \texttt{adapters.safetensors})$$

Checkpoint boyutu $\approx P_{LoRA} \times 4$ bayttır (float32 varsayımı). 1B ve r=8 için: $5\,636\,096 \times 4 \approx 22.5$ MB.

## 7. LoRA trainable parameter count ve yüzdesi

$$P_{LoRA} = n_{layers} \cdot r \cdot \sum_{m} (d_{in}^m + d_{out}^m), \qquad \%P = \frac{P_{LoRA}}{P_{total}} \times 100$$

1B için $\sum_m (d_{in}+d_{out}) = 44\,032$ (7 modül):

| r | n_layers | $P_{LoRA}$ | % (1 235.8 M içinde) |
|---|---|---|---|
| 4 | 8 | 1 409 024 | 0.114 % |
| 8 | 8 | 2 818 048 | 0.228 % |
| 8 | 16 | 5 636 096 | 0.456 % |
| 16 | 16 | 11 272 192 | 0.912 % |
| 32 | 16 | 22 544 384 | 1.824 % |

İlk dört satır kurulu MLX-LM ile ölçüldü **[Ölçüm]**. r=32 satırı aynı formülle hesaplandı.

Not: MLX-LM'in yazdırdığı yüzdenin paydası `get_total_parameters` değeridir. Bu fonksiyon quantized katmanları **orijinal parametre sayısıyla** sayar (`weight.size * 32 // bits`). Bu yüzden payda 4-bit modelde de 1 235.8 M'dir.

## 8. Model weight memory

$$M_{weights} = P_{total} \times \frac{\text{bit/parametre}}{8}$$

| Model | 16-bit | 4-bit (4.5 bit/param) |
|---|---|---|
| Llama 3.2 1B | 2.47 GB | 0.70 GB ([Ölçüm] dosya: 0.695 GB) |
| Llama 3.2 3B | 6.42 GB | 1.81 GB |
| Llama 3.1 8B | 16.06 GB | 4.52 GB |
| Mistral 7B v0.3 | 14.50 GB | 4.08 GB |

4-bit sütunu mlx-community repolarındaki gerçek dosya boyutlarıdır.

**Inference [Ölçüm]:** 1B 4-bit, kısa prompt ile peak memory 0.861 GB (backend'de 0.85–0.86 GB).

## 9. Training memory requirements (teorik)

$$M_{train} \approx M_{weights} + M_{LoRA} + M_{grad} + M_{Adam} + M_{act} + M_{overhead}$$

**LoRA kısmı** (1B, r=8, float32 varsayımı):

| Bileşen | Formül | Değer |
|---|---|---|
| LoRA ağırlıkları | $P \times 4$ B | 22.5 MB |
| Gradyanlar | $P \times 4$ B | 22.5 MB |
| Adam $m$, $v$ | $2P \times 4$ B | 45.1 MB |
| **Toplam** | | **~90 MB** |

Bu kısım küçüktür. Belleği asıl belirleyen **aktivasyonlardır**.

**Aktivasyonlar** (kaba üst sınır tahmini, bf16 = 2 bayt). Örnek: $B = 4$, $T = 512$, yani $BT = 2048$ token.

| Tensör | Formül | Değer |
|---|---|---|
| Katman girişi (checkpoint ile saklanan) | $L \cdot BT \cdot d \cdot 2$ | 16 × 2048 × 2048 × 2 = **134 MB** |
| MLP ara tensörleri (1 katman, gate/up/aktivasyon) | $\approx 3 \cdot BT \cdot d_{ff} \cdot 2$ | 3 × 2048 × 8192 × 2 = **101 MB** |
| Aynısı checkpoint **olmadan** tüm katmanlar için | $\times L$ | ≈ **1.6 GB** (+ attention ve norm tensörleri) |
| **Logits** | $BT \cdot V \cdot 4$ (fp32) | 2048 × 128 256 × 4 = **1.05 GB** |

Gözlemler:
- **Logit tensörü baskındır.** Llama 3'ün 128 256'lık vocab'ı yüzünden bu tensör $B \times T$ ile doğrusal büyür ve checkpoint'ten etkilenmez. $B = 4$, $T = 2048$ ile ~4.2 GB'a çıkar (fp32, teorik). Bu nedenle uzun örneklerde `batch_size` düşürmek en etkili bellek ayarıdır.
- `grad_checkpoint` katman içi ara tensörleri $L$ kat yerine yaklaşık 1 katman kadar tutar.
- Gerçek değer MLX'in kernel birleştirmelerine (fused SDPA), bellek havuzuna ve önbelleğine bağlıdır. **Ölçülmeden kesin kabul edilmemelidir.**

**[Referans] gerçek ölçüm:** Mistral 7B 16-bit, batch 8, örnek başına ~173 token, grad_checkpoint açık. Train peak memory 18.2 → 20.1 GB. Inference peak 14.7 GB. Bu konfigürasyon 16 GB'lık makinemize sığmaz.

**Bizim makine [Ölçüm]:** `max_recommended_working_set_size = 12.71 GB`. Bu değer eğitim için pratik tavandır.

## 10. Tahmini eğitim süresi

$$t_{train} \approx \frac{T_{işlenen}}{\text{throughput}_{tok/s}} + n_{eval} \cdot t_{eval}$$

Throughput bu makinede **ölçülmedi**. İlk eğitimden sonra log'daki `Tokens/sec` değeri formüle konacak.

**[Referans]** Mistral 7B 16-bit, bilinmeyen donanım: ~260–287 token/sn, 0.155–0.249 it/sn.

Aşağıdaki örnek **ölçüm değildir**, yalnızca formülün kullanımını gösterir. Varsayılan throughput: 1 000 token/sn.

$$T_{işlenen} = 600 \times 4 \times 300 = 720\,000 \text{ token} \Rightarrow t \approx 720 \text{ s} = 12 \text{ dk}$$

## 11. Loss ve perplexity ilişkisi

$$\mathrm{PPL} = e^{\mathcal{L}}, \qquad \mathcal{L} = \ln \mathrm{PPL}, \qquad \frac{\Delta\mathrm{PPL}}{\mathrm{PPL}} \approx \Delta\mathcal{L}$$

Loss'taki küçük bir mutlak düşüş, perplexity'de **oransal** bir düşüş demektir. Örneğin loss'ta 0.1'lik düşüş, PPL'de yaklaşık %10'luk düşüştür.

| Loss | PPL |
|---|---|
| 3.945 | 51.68 **[Referans]** test |
| 3.905 | 49.65 **[Referans]** ilk validation |
| 2.0 | 7.39 |
| 1.7 | 5.47 **[Referans]** ~190. adım train loss |
| 1.0 | 2.72 |

## 12. Senaryolar: 800, 1 000 ve 5 000 eğitim örneği

Ortak varsayımlar (**[Öneri]**, ölçülmedi): 1B 4-bit, r=8, num_layers=16, $\bar\ell = 300$ token (template dahil), $N_{val} = N_{train}/8$, `val_batches: -1`, `steps_per_eval = save_every`.

| | **A: 800** | **B: 1 000** | **C: 5 000** |
|---|---|---|---|
| batch_size $B$ | 4 | 4 | 4 |
| grad_accumulation_steps | 1 | 1 | 2 |
| $B_{eff}$ | 4 | 4 | 8 |
| $S_{epoch} = \lfloor N/B \rfloor$ | 200 | 250 | 1 250 |
| Dışarıda kalan örnek | 0 | 0 | 0 |
| Hedef epoch | 3 | 3 | 2 |
| **iters** | **600** | **750** | **2 500** |
| Optimizer güncellemesi $U$ | 600 | 750 | 1 250 |
| İşlenen örnek ($\text{iters} \times B$) | 2 400 | 3 000 | 10 000 |
| İşlenen token (≈) | 720 000 | 900 000 | 3 000 000 |
| $N_{val}$ | 100 | 125 | 625 |
| Validation batch $\lfloor N_{val}/B \rfloor$ | 25 (100 örnek) | 31 (124 örnek, 1 dışarıda) | 156 (624 örnek) |
| steps_per_eval = save_every | 50 | 75 | 250 |
| Validation ölçümü | 13 | 11 | 11 |
| Checkpoint | 12 + final | 10 + final | 10 + final |
| Checkpoint disk (22.5 MB) | ~293 MB | ~248 MB | ~248 MB |
| Süre @ 1 000 tok/s (**örnek**) | ~12 dk | ~15 dk | ~50 dk |

Senaryo B'de $N_{val} = 125$ değeri 4'e bölünmez, bu yüzden validation'da 1 örnek dışarıda kalır. Valid setini 124 veya 128 yapmak bu sorunu ortadan kaldırır.

Senaryo C'de 2 epoch seçildi. Büyük veride her örneği daha az tekrar etmek overfitting riskini azaltır. Validation eğrisi düşmeye devam ediyorsa üçüncü epoch eklenir.

### Referans projeye aynı hesapların uygulanması

| | Notebook komut hücresi | README / log konfigürasyonu |
|---|---|---|
| $N_{train}$ (~500 × 0.8) | ~400 | ~400 |
| $B$ | 16 | 8 |
| $S_{epoch}$ | 25 | 50 |
| iters | 200 | 1000 |
| **epoch** | **8.0** | **20.0** |
| Eğitmenin hesabı | 200 × 16 / 500 = 6.4 | — |

Eğitmenin 6.4 hesabı, toplam kayıt sayısını (500) train seti yerine kullandığı için düşük çıkmıştır.

---

## Kaynaklar

- MLX-LM 0.32.2 kaynak kodu: `tuner/trainer.py` (`iterate_batches`, `evaluate`, `train`), `tuner/utils.py` (`get_total_parameters`)
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016. arXiv:1604.06174
- Referans notebook hücre 13 (eğitim log'u) ve hücre 15 (test sonucu)
