---
language:
- tr
license: apache-2.0
task_categories:
- question-answering
- text-generation
tags:
- hukuk
- law
- turkish
- qa
size_categories:
- 10K<n<100K
---

# 📚 Turkish Law QA Dataset (Türkçe Hukuk Soru-Cevap Veri Seti)

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="300"/>
  <!-- İsterseniz buraya kendi logonuzu ekleyebilirsiniz -->
</div>

**Turkish Law QA Dataset**, Türk hukuku üzerine odaklanmış, çeşitli hukuki metinlerden, içtihatlardan ve mevzuatlardan titizlikle derlenmiş **18,300+** soru-cevap çiftinden oluşan kapsamlı bir veri setidir.

Bu veri seti, özellikle hukuk alanında uzmanlaşmış Büyük Dil Modellerini (LLM) ince ayarlamak (fine-tuning), RAG (Retrieval-Augmented Generation) sistemlerinin performansını test etmek ve Türk hukuk sistemine hakim yapay zeka asistanları geliştirmek amacıyla **OrionCAF** ekibi tarafından hazırlanmıştır.

## 📋 İçindekiler
* [Genel Açıklama](#-genel-açıklama)
* [İçerik ve Özellikler](#-içerik-ve-özellikler)
* [Hızlı Başlangıç](#-hızlı-başlangıç)
* [Veri Seti Detayları](#-veri-seti-detayları)
* [Ekibimiz](#-ekibimiz)
* [Yasal Uyarı](#-yasal-uyarı)
* [Lisans](#-lisans)

## ✨ İçerik ve Özellikler

Bu veri seti, ham hukuk verilerinin işlenerek "Soru" ve "Cevap" formatına dönüştürülmesiyle oluşturulmuştur.

*   **Kapsamlı Hukuk Bilgisi:** İstihkak Davası, Sebepsiz Zenginleşme, Limited Şirketler, Ticaret Hukuku gibi birçok kritik konuyu içerir.
*   **Temiz ve Yapılandırılmış:** Her kayıt `question` (soru) ve `answer` (cevap) alanlarından oluşan temiz bir JSON formatındadır.
*   **Yüksek Kalite:** Hukuki terminolojiye sadık kalınarak hazırlanmış yanıtlar.
*   **Çok Yönlü Kullanım:** 
    *   LLM Fine-tuning (Eğitim)
    *   RAG Benchmark (Test)
    *   Hukuki Chatbot Geliştirme

## 🚀 Hızlı Başlangıç

Bu veri setini Hugging Face `datasets` kütüphanesi ile saniyeler içinde indirebilir ve kullanmaya başlayabilirsiniz.

```python
from datasets import load_dataset

# Veri setini yükle
dataset = load_dataset("OrionCAF/turkish_law_qa_dataset")

# İlk örneği görüntüle
print(dataset['train'][0])

# Çıktı Örneği:
# {
#   'question': 'İstihkak davasının amacı nedir ve hangi durumlarda uygulanır?',
#   'answer': 'İstihkak davasının amacı, haksız olarak elde bulundurulan malın geri alınmasıdır...'
# }
```

## 🧠 Veri Seti Detayları

| Özellik | Değer |
| :--- | :--- |
| **Veri Seti Adı** | Turkish Law QA Dataset |
| **Yayınlayan** | OrionCAF |
| **Toplam Veri Sayısı** | 18,305 Çift |
| **Format** | JSON / Parquet |
| **Dil** | Türkçe (TR) |
| **Kaynaklar** | Türk Hukuk Mevzuatı, Yargıtay Kararları, Hukuki Makaleler |
| **Lisans** | Apache 2.0 |

## 👥 Ekibimiz

Bu veri seti, açık kaynak yapay zeka ekosistemine katkı sağlamak amacıyla **OrionCAF** ekibi tarafından geliştirilmiştir.

*   **Ferhat Kürkçüoğlu** – [LinkedIn](#)
*   **Cengizhan Bayram** – [LinkedIn](#)
*   **Cevdet Ahmet Turan** – [LinkedIn](#)

*Ekibe katılmak, katkı sağlamak veya geri bildirimde bulunmak için [Community](https://huggingface.co/OrionCAF/turkish_law_qa_dataset/discussions) sekmesini kullanabilirsiniz.*

## ⚠️ Yasal Uyarı

> [!IMPORTANT]
> Bu veri seti yalnızca eğitim ve araştırma amaçlıdır. İçerdiği bilgiler profesyonel hukuki tavsiye niteliği taşımaz. Hukuk, dinamik bir alandır ve mevzuat değişiklikleri veya içtihat farklılıkları olabilir. Veri setindeki bilgilerin kullanımı sonucunda doğabilecek herhangi bir sorumluluk kabul edilmez.

## 📄 Lisans

Bu proje **Apache License 2.0** altında lisanslanmıştır. Uygun atıf yapılarak akademik ve ticari projelerde serbestçe kullanılabilir.
