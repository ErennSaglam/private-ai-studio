# 00 · Technologies and Dependencies

[← README](../README.md)

Her teknoloji aynı başlıklarla anlatılır: **Nedir?**, **Neden kullanılır?**, **Hangi aşamada?**, **Zorunlu mu?**, **İlişkili bağımlılıklar** ve **Projemizde kullanılıyor mu?**

Sürüm bilgisi kurulu ortamdan okundu (24 Kasım 2026).

## Özet tablo

| Teknoloji | Projede | Sürüm | Aşama | Zorunluluk |
|---|---|---|---|---|
| Python | ✅ Kullanılıyor | 3.12.16 | Tümü | Zorunlu |
| pip | ✅ Kullanılıyor | venv içinde | Kurulum | Zorunlu (alternatif: uv) |
| venv | ✅ Kullanılıyor | stdlib | Kurulum | Güçlü öneri |
| MLX | ✅ Dolaylı | 0.32.4 | Inference, eğitim | Zorunlu (Apple Silicon yolu) |
| MLX-LM | ✅ Doğrudan | 0.32.2 | Inference, eğitim, fuse | Zorunlu |
| Hugging Face Hub | ✅ Dolaylı | 2.4.0 | Model indirme | İndirme için zorunlu |
| Transformers | ✅ Dolaylı | 5.24.0 | Tokenizer | MLX-LM içinde zorunlu |
| Safetensors | ✅ Format olarak | 0.8.0 | Ağırlık depolama | Zorunlu format |
| NumPy | ✅ Dolaylı | 2.5.5 | Eğitimde batch hazırlama | MLX-LM içinde zorunlu |
| FastAPI | ✅ Doğrudan | 0.165.0 | Servis | Zorunlu (alternatif: Flask) |
| uvicorn | ✅ Doğrudan | 0.54.0 | Servis | FastAPI için zorunlu |
| React | ✅ Doğrudan | 19.x | Arayüz | Seçim |
| Vite | ✅ Doğrudan | 8.3.x | Arayüz geliştirme | Seçim |
| Ollama | ❌ Kullanılmıyor | — | (Alternatif inference) | Alternatif |
| llama.cpp | ❌ Kullanılmıyor | — | (GGUF dönüşümü) | Alternatif |
| GGUF | ❌ Kullanılmıyor | — | (Alternatif format) | Alternatif |
| Open WebUI | ❌ Kullanılmıyor | — | (Alternatif arayüz) | Alternatif |

---

## Python

- **Nedir?** Genel amaçlı bir programlama dili. ML ekosisteminin ortak dilidir.
- **Neden?** MLX-LM, FastAPI ve Hugging Face araçları Python kütüphanesidir.
- **Aşama:** Tüm backend, eğitim ve değerlendirme.
- **Zorunlu mu?** Evet. `mlx-lm 0.32.2` ve `fastapi 0.165.0` Python ≥ 3.11 ister. macOS'un sistem Python'u 3.9.6 olduğu için Homebrew ile `python@3.12` kuruldu.
- **İlişkiler:** venv, pip ve bütün Python paketleri.
- **Projede:** ✅ `/opt/homebrew/bin/python3.12` (3.12.16).

## pip

- **Nedir?** Python paket yöneticisi. PyPI'dan paket indirip kurar.
- **Neden?** `requirements.txt` içindeki sabit sürümleri tekrarlanabilir şekilde kurmak için.
- **Aşama:** Kurulum.
- **Zorunlu mu?** Bir paket yöneticisi zorunludur. pip yerine `uv` veya `pip-tools` kullanılabilir.
- **İlişkiler:** venv içinde çalışır. Dolaylı bağımlılıkları (`transformers`, `numpy` vb.) kendisi çözer.
- **Projede:** ✅

## venv

- **Nedir?** Python'un yerleşik sanal ortam aracı. Projeye özel ve izole bir `site-packages` klasörü oluşturur.
- **Neden?** Paket sürümleri sisteme veya diğer projelere karışmaz. Java tarafındaki karşılığı, proje başına `pom.xml` ve ayrı bir bağımlılık deposu fikrine benzer.
- **Aşama:** Kurulum ve her çalıştırma (`source .venv/bin/activate`).
- **Zorunlu mu?** Teknik olarak hayır, pratikte evet. Alternatifler: conda, uv.
- **Projede:** ✅ `backend/.venv` (`.gitignore` içinde).

## MLX

- **Nedir?** Apple'ın Apple Silicon için geliştirdiği array ve makine öğrenmesi framework'ü. API'si NumPy ve PyTorch'a benzer.
- **Neden?** Apple Silicon'da CPU ve GPU aynı fiziksel belleği paylaşır (**unified memory**). MLX bu yapıya göre tasarlanmıştır: tensörler CPU ile GPU arasında kopyalanmaz. Hesaplamalar **lazy** çalışır; `mx.eval` çağrılana kadar yalnızca hesaplama grafiği kurulur. GPU erişimi Metal üzerinden yapılır.
- **Aşama:** Inference ve eğitim. Matris çarpımı, quantized matmul ve otomatik türev hesabı (`mx.grad`, `nn.value_and_grad`) bu katmanda yapılır.
- **Zorunlu mu?** Apple Silicon yolunu seçtiğimiz için evet. Alternatifler: PyTorch (MPS backend), llama.cpp.
- **İlişkiler:** `mlx-metal` (GPU çekirdekleri). MLX-LM bunun üzerine kuruludur.
- **Projede:** ✅ Dolaylı bağımlılık (0.32.4). `inspect_model.py` doğrudan `mlx.core` kullanır. `model_service.py` ise GPU stream'i oluşturmak için `mx.new_stream` çağırır.
- **Bizim karşılaştığımız davranış:** GPU stream thread'e bağlıdır. Bu yüzden servis tüm MLX çağrılarını tek bir worker thread'de yapar.

## MLX-LM

- **Nedir?** MLX üzerinde LLM çalıştırma ve eğitme paketi.
- **Neden?** Tek pakette şunları verir:

| Komut / API | Görev |
|---|---|
| `load()` | Model ve tokenizer'ı yükler, isteğe bağlı olarak adapter da yükler |
| `stream_generate()`, `generate()` | Token üretimi |
| `mlx_lm.generate` | Terminalden üretim |
| `mlx_lm.lora` | LoRA, DoRA veya full fine-tuning; `--test` ile test loss ve perplexity |
| `mlx_lm.fuse` | Adapter'ı base modele birleştirme, isteğe bağlı GGUF export |
| `mlx_lm.convert` | Hugging Face modelini MLX formatına çevirme ve quantize etme |
| `mlx_lm.evaluate`, `mlx_lm.perplexity`, `mlx_lm.benchmark` | Değerlendirme ve ölçüm araçları |
| `mlx_lm.server` | OpenAI uyumlu HTTP sunucusu (biz kendi FastAPI'mizi kullanıyoruz) |

- **Aşama:** Faz 1 (inference), Faz 5–8 (eğitim ve değerlendirme).
- **Zorunlu mu?** Evet.
- **İlişkiler:** mlx, transformers (tokenizer), numpy, jinja2 (chat template), sentencepiece, protobuf, pyyaml (YAML config).
- **Projede:** ✅ `requirements.txt` içinde `mlx-lm==0.32.2`.

## Hugging Face Hub

- **Nedir?** Model ve veri seti deposu. `huggingface_hub` ise bu depoya erişen Python kütüphanesi ve `hf` CLI'dır.
- **Neden?** Modeli bir kez indirmek için.
- **Aşama:** Yalnızca kurulum. Çalışma zamanında `HF_HUB_OFFLINE=1` ile kapatılır.
- **Zorunlu mu?** İndirme için evet. Ağırlıklar başka bir yoldan kopyalanırsa (örneğin USB ile kurumsal ortama) gerekmez.
- **İlişkiler:** `transformers` tarafından kurulur.
- **Projede:** ✅ 2.4.0. `hf download … --local-dir models/…` kullanıldı. Model gizli cache yerine proje klasöründe durur.

## Transformers

- **Nedir?** Hugging Face'in model ve tokenizer kütüphanesi.
- **Neden?** MLX-LM, tokenizer'ı (`tokenizer.json`) ve Jinja chat template'i bu kütüphane ile yükler. Model hesaplaması MLX'te yapılır, Transformers'ta değil.
- **Aşama:** Inference ve eğitim (tokenization ve chat template).
- **Zorunlu mu?** MLX-LM içinde zorunlu.
- **Projede:** ✅ Dolaylı (5.24.0).

## Safetensors

- **Nedir?** Tensörleri saklamak için güvenli ve hızlı bir dosya formatı. Başlıkta JSON metadata, ardından ham tensör baytları bulunur.
- **Neden?** PyTorch'un `.bin` (pickle) formatı yüklenirken keyfi kod çalıştırabilir. Safetensors yalnızca veri içerir. Ayrıca memory-map ile hızlı yüklenir.
- **Aşama:** Model ağırlıkları ve gelecekteki LoRA adapter dosyaları (`adapters.safetensors`).
- **Zorunlu mu?** MLX-LM'in kullandığı formattır.
- **Projede:** ✅ `model.safetensors` (663 MB, 372 tensör).

## NumPy

- **Nedir?** Python'un temel sayısal hesaplama kütüphanesi.
- **Neden?** MLX-LM eğitimde batch'leri NumPy ile hazırlar (`iterate_batches` içinde `np.random.permutation` ve padding). `--seed` hem `mx.random.seed` hem `np.random.seed` için kullanılır.
- **Aşama:** Eğitim.
- **Projede:** ✅ Dolaylı (2.5.5). Bizim kodumuz doğrudan kullanmaz.

## FastAPI

- **Nedir?** Python için tip ipuçlarına dayalı bir web framework'ü. Spring Boot'taki `@RestController` ile benzer görevi görür.
- **Neden?** Pydantic ile otomatik istek doğrulama, otomatik OpenAPI ve Swagger dokümantasyonu (`/docs`), ve modeli açılışta bir kez yüklemek için `lifespan` desteği.
- **Aşama:** Servis katmanı (Faz 1 ve sonrası).
- **Zorunlu mu?** Bir HTTP katmanı zorunlu. FastAPI bir seçimdir (alternatifler: Flask, Litestar, `mlx_lm.server`).
- **İlişkiler:** pydantic, starlette. uvicorn ile çalışır.
- **Projede:** ✅ 0.165.0.

## uvicorn

- **Nedir?** ASGI web sunucusu. Spring Boot'taki gömülü Tomcat ile benzer görevi görür.
- **Neden?** FastAPI uygulamasını bir porttan dinlemeye açar.
- **Projede:** ✅ 0.54.0, `--host 127.0.0.1 --port 3001`.

## React

- **Nedir?** Bileşen tabanlı bir UI kütüphanesi.
- **Neden?** Sohbet arayüzü, durum yönetimi (`useState`, `useEffect`).
- **Zorunlu mu?** Hayır, bir seçimdir.
- **Projede:** ✅ 19.x, [frontend/src/App.jsx](../frontend/src/App.jsx).

## Vite

- **Nedir?** Frontend geliştirme sunucusu ve build aracı.
- **Neden?** Hızlı geliştirme sunucusu ve production build (`npm run build`).
- **Projede:** ✅ 8.3.x. `npm create vite@latest` ile oluşturuldu (create-vite 9.3.4). Port 3000 ve `strictPort: true`; böylece port CORS ayarıyla her zaman uyuşur.

---

## Projede kullanılmayan teknolojiler

### Ollama

- **Nedir?** GGUF modelleri yerelde çalıştıran bir sunucu ve CLI. Model, sampling parametreleri ve chat template bir `Modelfile` ile tanımlanır.
- **Neden kullanılır?** Kurulumu kolaydır ve hazır bir REST API sunar. Open WebUI gibi arayüzlerle hemen çalışır.
- **Hangi aşamada?** Referans projede eğitim **sonrası** dağıtım adımında: fuse → GGUF → `ollama create`.
- **Zorunlu mu?** Hayır, alternatif bir dağıtım yoludur.
- **İlişkiler:** llama.cpp (Ollama'nın motoru), GGUF.
- **Projede:** ❌ Kullanılmıyor. Bizim inference yolumuz MLX-LM + FastAPI.

### llama.cpp

- **Nedir?** C/C++ ile yazılmış, CPU, Metal ve CUDA destekli bir LLM inference motoru. `convert_hf_to_gguf.py` dönüştürücüsünü içerir.
- **Neden kullanılır?** Taşınabilirlik ve çok sayıda quantization seçeneği (Q4_K_M, Q8_0 vb.). Apple dışı donanımda da çalışır.
- **Hangi aşamada?** Referans projede fuse edilmiş modeli `--outtype q8_0` ile GGUF'a çevirmek için.
- **Zorunlu mu?** Hayır. MLX-LM 0.32.2'de `mlx_lm.fuse --export-gguf` seçeneği de var (kurulu sürümün `--help` çıktısında görüldü; projede denenmedi).
- **Projede:** ❌ Kullanılmıyor.

### GGUF

- **Nedir?** llama.cpp ekosisteminin tek dosyalık model formatı. Ağırlıklar, tokenizer ve metadata aynı dosyadadır.
- **Neden kullanılır?** Ollama ve llama.cpp tabanlı araçlar için gereklidir.
- **Projede:** ❌ Kullanılmıyor. Bizim formatımız MLX safetensors ve ayrı `config.json` / `tokenizer.json` dosyaları.

### Open WebUI

- **Nedir?** Ollama veya OpenAI uyumlu API'lere bağlanan, kendi sunucunda çalışan bir web arayüzü.
- **Neden kullanılır?** Hazır bir sohbet arayüzü ve kullanıcı yönetimi sunar.
- **Projede:** ❌ Kullanılmıyor. Kendi React arayüzümüzü yazdık, çünkü eğitim ve değerlendirme süreçlerini (adapter seçimi, metrikler) arayüze eklemeyi planlıyoruz.

---

## Alternatif yollar: neden MLX-LM?

| Ölçüt | MLX-LM (seçilen) | Ollama + llama.cpp | PyTorch + PEFT (MPS) |
|---|---|---|---|
| Apple Silicon optimizasyonu | Yerel (Metal, unified memory) | İyi (Metal) | Kısıtlı (MPS backend) |
| Aynı araçla eğitim + inference | Evet (`mlx_lm.lora`) | Hayır, yalnızca inference | Evet |
| QLoRA (quantized base üzerinde LoRA) | Evet | — | `bitsandbytes` Apple'da sınırlı |
| Python API ile servis entegrasyonu | Doğrudan (`load`, `stream_generate`) | HTTP üzerinden | Doğrudan |
| Apple dışı donanıma taşınabilirlik | Hayır | Evet | Evet |

Taşınabilirlik notu: Şirket sunucuları genellikle Linux + NVIDIA'dır. Bu projede öğrenilen kavramlar (LoRA, dataset, evaluation) taşınabilir. MLX'e özgü kod (`mlx_lm.lora`) ise Linux'ta PEFT veya benzeri bir araçla değiştirilmelidir. LoRA adapter'ları framework'ler arasında doğrudan uyumlu değildir; dönüşüm gerekir. Referans repodaki `Safetensors_to_bin.py` bu yönde bir girişimdir.
