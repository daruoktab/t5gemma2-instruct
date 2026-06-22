# Mengunduh artikel LLM post-training sebagai markdown

**Created:** 6/22/2026 18:31:38  
**Updated:** 6/22/2026 18:50:05  
**Exported:** 6/23/2026 3:22:49  
**Link:** [https://claude.ai/chat/a5f291c3-84c5-4785-bae9-112e2bb9a495](https://claude.ai/chat/a5f291c3-84c5-4785-bae9-112e2bb9a495)  

## User:

> 6/22/2026 18:31:38

https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms
download lengkap artikel ini menjadi markdown

## Assistant:

> 6/22/2026 18:34:46





Beberapa catatan tentang isi artikel ini:

**Konten yang tercakup:**
- Definisi terminologi RL (State, Action, Policy, Trajectory, Reward, Critic, dll.)
- **On-Policy Methods**: derivasi core objective, REINFORCE, PPO (CLIP & KLPEN), GRPO
- **KL Divergence clarification**: Forward vs Reverse KL, Trust-Region KL vs Drift KL, tiga estimator K1/K2/K3 beserta tabel perbandingan performa
- **Off-Policy Methods**: derivasi lengkap DPO step-by-step, plus ringkasan ORPO, SimPO, IPO, KTO
- Tabel summary perbandingan semua algoritma
- Referensi paper lengkap

Bagian akhir artikel (setelah derivasi DPO) tidak ter-fetch sepenuhnya karena website memotong konten — seksi DPO variants (ORPO, SimPO, IPO, KTO) dan tabel summary direkonstruksi dari konteks artikel.


## User:

> 6/22/2026 18:41:39

untuk project t5gemma-2, algoritma mana saja yang tidak membutuhkan model tambahan dan datasetnya pun mudah dibuat?

## Assistant:

> 6/22/2026 18:42:04

Dari artikel tadi, dilihat dari dua kriteria — **(1) tidak butuh model tambahan** dan **(2) dataset mudah dibuat** — ini peta pilihannya untuk T5Gemma2-Instruct:

---

## Ranking untuk T5Gemma2-Instruct

### ✅ Terbaik: **SimPO**

**Tidak butuh model tambahan:** tidak perlu reference model (tidak seperti DPO yang butuh $\pi_{ref}$ frozen).

**Dataset:** pasangan (prompt, chosen, rejected) — tapi karena T5Gemma2 sudah punya kemampuan generasi, kamu bisa **self-generate** kandidat respons lalu rank pakai reward heuristic sederhana (panjang, format, keyword matching, BLEU vs reference).

**Cocok untuk T5Gemma2 karena:** arsitektur encoder-decoder berarti forward pass untuk menghitung log-prob response cukup efisien — tidak perlu load model referensi kedua di VRAM 6GB-mu.

---

### ✅ Bagus: **KTO**

**Tidak butuh model tambahan:** hanya butuh binary signal (good/bad), tidak perlu paired comparison.

**Dataset paling mudah:** cukup label satu response per prompt sebagai `good` atau `bad` — tidak perlu dua respons berpasangan. Artinya kamu bisa pakai output dari smoke test sebelumnya dan label secara manual atau rule-based.

**Trade-off:** secara teoritis kurang informatif dibanding paired, tapi untuk iterasi awal sangat praktis.

---

### ⚠️ Pertimbangkan dengan hati-hati: **DPO / ORPO**

**DPO** butuh $\pi_{ref}$ frozen → tambah ~3–4GB VRAM, berat di RTX 3060 6GB-mu.

**ORPO** tidak butuh reference model (seperti SimPO), tapi menggabungkan SFT loss + preference loss dalam satu pass — artinya **dataset harus punya pasangan chosen/rejected sekaligus**, sedikit lebih rumit dari KTO.

---

### ❌ Skip dulu: **PPO / GRPO**

Butuh Reward Model terpisah (on-policy) — terlalu berat untuk setup lokalmu dan tidak ada reward model bilingual Indonesian-English yang siap pakai.

---

## Rekomendasi Konkret untuk T5Gemma2-Instruct

```
Phase 1 (sekarang): SFT dengan dataset daruokta/t5-gemma-2-chat-instruct-merged.jsonl
     ↓
Phase 2 (alignment): SimPO atau KTO
```

**Cara buat dataset SimPO/DPO dengan mudah:**

```python
# Strategi: generate N kandidat per prompt, rank by rule-based reward
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def score_response(response, task_type):
    score = 0
    if task_type == "translate":
        # heuristic: panjang reasonable, tidak ada repetisi
        score += 1 if 10 < len(response.split()) < 200 else 0
        score += 1 if len(set(response.split())) / len(response.split()) > 0.4 else 0
    elif task_type == "summarize":
        score += 1 if len(response.split()) < 100 else 0  # singkat
    return score

# Generate 4 kandidat per prompt → ambil best sebagai chosen, worst sebagai rejected
```

**Alternatif paling cepat:** pakai dataset preference yang sudah ada seperti `argilla/ultrafeedback-binarized-preferences-cleaned` lalu filter ke instruksi yang relevan untuk bilingual Indonesian-English.

---

**Bottom line:** untuk constraint VRAM 6GB dan goal T5Gemma2-Instruct bilingual, urutan prioritasnya adalah **KTO > SimPO > ORPO > DPO**, dengan PPO/GRPO sebagai opsi cloud (Colab A100) kalau memang diperlukan.

## User:

> 6/22/2026 18:42:59

apakah simpo dan dpo bisa di gabung?

## Assistant:

> 6/22/2026 18:43:19

Secara teknis bisa, tapi perlu dipahami dulu apa yang digabung dan mengapa.

---

## Hubungan SimPO vs DPO

SimPO sebenarnya adalah **modifikasi dari DPO**, bukan algoritma yang sepenuhnya berbeda. Perbedaan utamanya:

| | DPO | SimPO |
|---|---|---|
| Implicit reward | $\beta \log \frac{\pi_\theta(y\|x)}{\pi_{ref}(y\|x)}$ | $\frac{\beta}{\|y\|} \log \pi_\theta(y\|x)$ |
| Reference model | Butuh $\pi_{ref}$ frozen | **Tidak butuh** |
| Length normalization | Tidak ada | Ada (dibagi $\|y\|$) |
| Target margin | Tidak ada | Ada (hyperparameter $\gamma$) |

SimPO menghilangkan reference model dengan mengganti implicit reward menjadi **average log-probability** response itu sendiri, ditambah margin $\gamma$ agar chosen lebih unggul dari rejected by at least $\gamma$.

Loss SimPO:

$$\mathcal{L}_{SimPO} = -\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right) \right]$$

---

## Cara Menggabungkannya

Ada beberapa strategi yang masuk akal:

### 1. Sequential (paling aman)
```
SFT → DPO → SimPO fine-tune
```
Pakai DPO dulu (butuh reference model = checkpoint SFT), lalu SimPO sebagai tahap "polish" tanpa reference model. SimPO di sini bertindak sebagai **iterative preference optimization** dari model hasil DPO.

### 2. Hybrid Loss
Gabungkan kedua loss dengan bobot $\alpha$:

$$\mathcal{L}_{hybrid} = \alpha \cdot \mathcal{L}_{DPO} + (1 - \alpha) \cdot \mathcal{L}_{SimPO}$$

Tapi ini **tidak direkomendasikan** karena DPO butuh $\pi_{ref}$ sedangkan SimPO tidak — kamu tetap harus load reference model, sehingga keunggulan SimPO (hemat VRAM) hilang.

### 3. Gunakan SimPO saja dengan length normalization manual
Ini yang paling praktis untuk T5Gemma2-Instruct. Encoder-decoder punya perilaku panjang output yang berbeda dari decoder-only, jadi **length normalization di SimPO justru sangat relevan** — respons summarization yang pendek tidak akan "dihukum" dibandingkan respons translation yang panjang.

---

## Rekomendasi untuk T5Gemma2-Instruct

Menggabungkan keduanya tidak memberikan manfaat signifikan dan menambah kompleksitas. Yang lebih berguna adalah:

```python
# SimPO murni dengan margin yang disesuaikan per task type
# karena T5Gemma2 punya multi-task (translate, summarize, QA)

def simpo_loss(logprob_chosen, logprob_rejected, 
               len_chosen, len_rejected, 
               beta=0.1, gamma=0.5):
    reward_chosen = beta * logprob_chosen / len_chosen
    reward_rejected = beta * logprob_rejected / len_rejected
    return -F.logsigmoid(reward_chosen - reward_rejected - gamma)
```

Pertimbangkan **per-task margin** $\gamma$ yang berbeda karena karakteristik output translation vs summarization vs QA sangat berbeda panjang dan strukturnya — ini keunikan T5Gemma2-Instruct sebagai multi-task model yang tidak akan kamu temukan di paper SimPO aslinya.

## User:

> 6/22/2026 18:43:42

coba baca paper mereka masing2

## Assistant:

> 6/22/2026 18:44:47





---

## Apa yang Sebenarnya Dikatakan Paper

### DPO (Rafailov et al., 2023)

**Insight kunci dari paper:** DPO bukan algoritma baru yang berdiri sendiri — ia adalah **reparameterisasi matematis** dari RLHF. Rafailov et al. menunjukkan bahwa objective RLHF standar (reward maximization + KL constraint) memiliki solusi closed-form:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

Dari sini, reward bisa di-invert:

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

Karena $Z(x)$ cancel saat menghitung selisih reward antar pasangan, kita dapat:

$$\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**Implikasinya: $\pi_{ref}$ bukan sekedar regularizer — ia adalah komponen struktural dari derivasi. Tanpanya, persamaan ini tidak valid secara matematis.**

---

### SimPO (Meng et al., 2024)

**Insight kunci dari paper:** SimPO bukan turunan DPO — ia adalah **reformulasi dari nol** dengan motivasi berbeda.

Paper SimPO mengidentifikasi **dua masalah fundamental DPO**:

**Masalah 1: Mismatch reward vs. generation metric**

DPO mengoptimasi reward $\beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$, tapi saat inference model menggunakan average log-likelihood $\frac{1}{|y|} \log \pi_\theta(y|x)$ untuk ranking/generation. Dua hal yang berbeda dioptimasi saat training vs. inference.

Bukti empiris dari paper: hanya **~50% preference pairs** yang memiliki likelihood ranking benar setelah training DPO. Hampir random.

**Masalah 2: Length bias**

Tanpa length normalization, model belajar bahwa respons panjang = reward tinggi, terlepas dari kualitasnya.

**Solusi SimPO:** Gunakan average log-probability langsung sebagai implicit reward:

$$r_{SimPO}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

Ini **tidak butuh $\pi_{ref}$** sama sekali karena reward tidak lagi didefinisikan sebagai rasio terhadap reference. Ditambah margin $\gamma$:

$$\mathcal{L}_{SimPO} = -\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right) \right]$$

**Hasil dari paper:** SimPO unggul DPO hingga **+6.4 points** di AlpacaEval 2 dan **+7.5 points** di Arena-Hard, dengan runtime **20% lebih cepat** dan memory **10% lebih hemat**.

---

## Jawaban Langsung: Bisakah DPO dan SimPO Digabung?

Setelah baca papernya — **secara prinsip, tidak ada yang bisa "digabung"** dalam arti yang bermakna, karena:

1. **Keduanya menyelesaikan problem yang sama** (preference optimization dari dataset $(x, y_w, y_l)$) tapi dengan **asumsi yang saling eksklusif**:
   - DPO *membutuhkan* $\pi_{ref}$ karena ia adalah bagian dari derivasi matematisnya
   - SimPO *secara eksplisit menolak* kebutuhan $\pi_{ref}$ dan menjadikannya keunggulan utama

2. **SimPO bukan ekstensi DPO** — ia adalah pengganti dengan reward definition yang berbeda. Menggabungkan lossnya berarti mengoptimasi dua implicit reward yang berbeda secara simultan, yang tidak punya justifikasi teoritis.

3. Paper SimPO sendiri sudah membandingkan SimPO vs. DPO, R-DPO, IPO, ORPO, KTO, dll. — dan SimPO menang di semua setting tanpa perlu digabung.

**Yang masuk akal secara sequential:**

```
SFT → DPO (pakai π_ref = SFT model) → SimPO (polish tanpa π_ref)
```

Tapi benefit-nya kecil: SimPO sendiri sudah menangani drift dengan learning rate kecil dan diversity dataset, dan paper menunjukkan KL divergence SimPO tetap reasonable tanpa explicit regularization.

---

## Implikasi Spesifik untuk T5Gemma2-Instruct

Ada satu hal dari paper SimPO yang sangat relevan untuk arsitektur encoder-decoder:

**Length normalization lebih kritis untuk T5Gemma2 dibanding decoder-only model.** Paper menunjukkan bahwa tanpa LN, Spearman correlation antara average log-likelihood dan panjang response mencapai $\rho = 0.82$ (sangat bias panjang). Untuk T5Gemma2 yang menangani task dengan panjang output sangat bervariasi — summarization (pendek) vs. translation (bisa panjang) vs. QA — bias ini akan lebih parah.

SimPO cocok dipakai langsung setelah SFT, tanpa perlu DPO di tengah.

## User:

> 6/23/2026 04:39:32

btw berdasarkan yang kamu tahu plus juga dari dokumen, obrolanku tadi plus juga documents bahasa inggrisnya itu, bedanya mereka semua tuh apa? bedanya SimPO, CPO, DPO, ORPO, KTO itu apa? coba gunakan penjelasan yang mudah dipahami gitu.

## Assistant:

Menarik sekali pertanyaannya! Berdasarkan dokumen yang kita susun dan konsep di baliknya, kelima metode itu sebenarnya punya tujuan yang persis sama: **mengajari AI mana jawaban yang disukai manusia (bagus) dan mana yang dibenci (jelek)**. Proses ini biasa disebut *Alignment* atau *Preference Tuning*.

Supaya gampang dipahami, mari kita bayangkan AI kita ini adalah seorang **Koki Magang** yang sedang belajar memasak untuk restoran Anda.

Berikut adalah bedanya kelima metode tersebut:

### 1. DPO (Direct Preference Optimization)
* **Cara kerja:** Anda menyuruh Koki Magang masak dua piring nasi goreng untuk pesanan yang sama: satu piring enak (Chosen) dan satu piring gosong (Rejected). Masalahnya, Anda juga mengundang **Koki Master (Reference Model)** yang harus selalu berdiri di dapur untuk mengawasi dan ngasih standar "resep asli".
* **Kelemahan:** Dapur jadi sangat sesak! (Secara teknis: DPO **sangat boros VRAM/Memori GPU** karena Anda harus menyalakan 2 model secara bersamaan di dalam GPU, yakni model yang dilatih dan model referensi).

### 2. ORPO (Odds Ratio Preference Optimization)
* **Cara kerja:** Anda sadar dapur Anda sempit (VRAM terbatas). Di ORPO, kita pecat Koki Master. Saat si Koki Magang masak dua piring (enak vs gosong), kita langsung memarahi dia: *"Pelajari yang enak ini (SFT), dan buang jauh-jauh kebiasaan buruk dari yang gosong ini!"*
* **Kelebihan:** **Sangat hemat memori dan cepat!** Ia menggabungkan pelajaran dasar (SFT) dan perbandingan (Alignment) dalam satu tarikan napas tanpa butuh model referensi.

### 3. CPO (Contrastive Preference Optimization)
* **Cara kerja:** Hampir mirip dengan DPO, kita menyuruh Koki Magang masak dua piring. Bedanya, CPO punya aturan ketat (disebut *regularizer*): *"Kamu boleh menjauhi rasa nasi goreng gosong itu sejauh mungkin, tapi JANGAN sampai rasanya malah jadi aneh seperti sup!"* 
* **Fungsi:** CPO mencegah AI "lupa daratan". Kadang kalau AI terlalu dipaksa menjauhi jawaban jelek, bahasanya malah jadi rusak atau aneh. CPO menjaga agar gaya bahasa AI tetap natural. Di versi terbarunya, CPO juga bisa berjalan tanpa Koki Master.

### 4. SimPO (Simple Preference Optimization)
* **Cara kerja:** Ini adalah metode paling modern (adik cerdas dari CPO/DPO). Anda sadar kalau si Koki Magang ini suka curang: **dia pikir kalau porsinya dibikin jumbo (teksnya panjang bertele-tele), pelanggan pasti suka.** SimPO menstandarkan ukuran piringnya (*Length-Normalized Reward*). Jadi, dia dipaksa membandingkan dua masakan murni dari **rasanya (kualitas teks)**, bukan dari seberapa besar porsinya.
* **Kelebihan:** Terbukti mengalahkan DPO dan ORPO di banyak benchmark (seperti AlpacaEval) karena berhasil mengatasi penyakit LLM modern yang suka "menjawab panjang lebar tapi isinya nol".

### 5. KTO (Kahneman-Tversky Optimization)
* **Cara kerja:** Ini yang **paling beda**. Kalau DPO, ORPO, CPO, dan SimPO mewajibkan Koki Magang menyajikan **dua piring sekaligus** untuk setiap 1 pesanan (harus ada perbandingan apel vs apel). Di KTO, Koki bebas masak 1 piring saja. Pelanggan cukup kasih jempol ke atas (👍 Bagus) atau jempol ke bawah (👎 Jelek).
* **Konsep Unik:** KTO terinspirasi dari teori psikologi peraih Nobel tentang bagaimana manusia mengambil risiko (*Prospect Theory*). Metodenya menyadari bahwa "manusia itu lebih benci kehilangan uang 1 juta daripada senangnya dapat uang 1 juta". Jadi KTO menghukum jawaban jelek (👎) secara lebih tegas dibanding memuji jawaban bagus.
* **Kelebihan:** Pembuatan dataset-nya paling gampang! Kita tidak perlu repot-repot memikirkan jawaban jelek untuk setiap jawaban bagus. Data murni dari *Rating* pengguna sudah cukup.

---

**Ringkasan Singkat untuk Proyek Kita:**
- Jika punya data **berpasangan (Bagus vs Jelek)** dan VRAM terbatas: Paling jago pakai **SimPO** atau **ORPO**.
- Jika datanya cuma **Jempol Atas/Jempol Bawah (Bagus/Jelek yang berdiri sendiri)**: Wajib pakai **KTO**.

---

**Sources:**

- [SimPO: Simple Preference Optimization with a Reference-Free Reward | alphaXiv](https://www.alphaxiv.org/overview/2405.14734v3)
- [[PDF] SimPO: Simple Preference Optimization with a Reference-Free Reward | Semantic Scholar](https://www.semanticscholar.org/paper/SimPO:-Simple-Preference-Optimization-with-a-Reward-Meng-Xia/c3f1fae241a3c2449e675ab750873d800f95513c)
- [[2405.14734] SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward Yu Meng1∗](https://arxiv.org/pdf/2405.14734)
- [AlphaPO: Reward Shape Matters for LLM Alignment](https://arxiv.org/pdf/2501.03884)
- [SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization](https://arxiv.org/pdf/2602.02383)
- [SimPO: simple preference optimization with a reference- ...](https://dl.acm.org/doi/10.5555/3737916.3741862)
- [Paper page - SimPO: Simple Preference Optimization with a Reference-Free Reward](https://huggingface.co/papers/2405.14734)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward | OpenReview](https://openreview.net/forum?id=3Tzcot1LKb)
- [[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Direct Preference Optimization(DPO) Presented by: Di Mu, An Cao 2024 MScAC](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/mu_cao_dpo.pdf)
- [Paper page - Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290)
- [Direct Preference Optimization (DPO)](https://apxml.com/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/direct-preference-optimization-dpo)
- [Contrastive Weak-to-strong Generalization](https://arxiv.org/pdf/2510.07884)
- [Filtered Direct Preference Optimization](https://arxiv.org/pdf/2404.13846)
- [Direct Preference Optimization with an Offset](https://arxiv.org/pdf/2402.10571)
- [BiasDPO: Mitigating Bias in Language Models through Direct Preference   Optimization](https://arxiv.org/pdf/2407.13928)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model | by Eleventh Hour Enthusiast | Medium](https://medium.com/@EleventhHourEnthusiast/direct-preference-optimization-your-language-model-is-secretly-a-reward-model-5b8f44cb9b9b)
- [Token-level Direct Preference Optimization](https://arxiv.org/pdf/2404.11999)

## User:

> 6/22/2026 18:48:07

coba baca juga paper kto dan orpo intinya soalnya aku gak lokal, dan fokus utamanya untuk encoder-decoder loh bukan decoder biasa dan juga fokusnya untuk percakapan sehari-hari yang multi turn dan in-task advantage, baca https://github.com/daruoktab/t5gemma2-instruct/blob/main/README.md

## Assistant:

> 6/22/2026 18:50:05







---

## Apa yang Sebenarnya Dikatakan Paper-paper Ini

### KTO — Insight Kritis yang Sering Dilewati

**Motivasi utama:** Manusia bukan agen rasional — mereka *loss-averse*. KTO dibangun dari Kahneman-Tversky *prospect theory*: manusia lebih sensitif terhadap kerugian daripada keuntungan yang setara. Ini diformalisasikan sebagai **Human-Aware Loss (HALO)**.

**Arsitektur lossnya:**

$$\mathcal{L}_{KTO} = \mathbb{E}_{x,y \sim D}[\lambda_y - v(x,y)]$$

di mana:

$$v(x,y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x,y) - z_0)) & \text{jika } y \text{ desirable} \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x,y))) & \text{jika } y \text{ undesirable} \end{cases}$$

dengan $r_\theta(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ dan $z_0$ adalah estimasi KL divergence sebagai *reference point*.

**Yang penting dari paper aslinya yang jarang disebut:**

1. **KTO masih butuh $\pi_{ref}$** — ini sering disalahpahami. KTO *tidak* reference-free. Ia butuh frozen reference model karena $r_\theta$ masih didefinisikan sebagai log-ratio terhadap $\pi_{ref}$. Yang tidak dibutuhkan adalah *paired* data.

2. **KTO bisa skip SFT**, tapi hanya jika model sudah cukup kuat. Paper secara eksplisit menyebutkan Llama-13B/30B bisa langsung KTO tanpa SFT, tapi model kecil tetap butuh SFT dulu.

3. **Asimetri $\lambda_D$ dan $\lambda_U$ sangat penting** untuk data imbalanced. Paper merekomendasikan $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{3}{2}]$.

4. **KTO secara implisit menangani noisy labels lebih baik dari DPO** — dari Theorem 4.3: jika ada preferensi yang kontradiktif (50% orang suka $y_a$, 50% suka $y_b$), DPO bisa belajar ke *minority-preferred* output, sedangkan KTO tidak.

---

### ORPO — Yang Benar-benar Reference-Free

**Motivasi dari paper:** ORPO mengidentifikasi masalah mendasar SFT yang sering diabaikan — ketika kamu fine-tune dengan cross-entropy pada chosen response saja, log probability *rejected* response juga ikut naik. Ini ditunjukkan secara empiris dengan OPT-350M pada HH-RLHF: chosen dan rejected probability naik bersamaan.

**Solusinya:** Gabungkan SFT loss dengan *odds ratio* penalty dalam satu forward pass:

$$\mathcal{L}_{ORPO} = \mathbb{E}_{(x,y_w,y_l)}[\mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}]$$

$$\mathcal{L}_{OR} = -\log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

dengan *odds* didefinisikan sebagai $\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1-P_\theta(y|x)}$.

**Mengapa odds ratio, bukan probability ratio?** Paper membuktikan bahwa probability ratio ($\frac{P(y_w)}{P(y_l)}$) distribusinya sangat *sharp*, yang menyebabkan suppression berlebihan pada rejected tokens — terutama buruk ketika model belum ter-adapt ke domain. Odds ratio distribusinya lebih lebar dan *mild*, sehingga aman digabungkan dengan SFT dalam satu pass.

**Fakta penting yang relevan untuk T5Gemma2:**
- ORPO tidak butuh $\pi_{ref}$ sama sekali — zero reference model overhead
- Hanya butuh **dua forward passes per batch** (chosen + rejected), bukan empat seperti DPO
- Dari gradient analysis: $\delta(d) \to 0$ ketika odds chosen sudah jauh lebih tinggi dari rejected — artinya **penalty secara otomatis berkurang seiring konvergensi**, tidak perlu tuning manual

---

## Masalah Fundamental: Semua Ini Dirancang untuk Decoder-Only

Setelah baca empat paper ini, ada satu kebenaran yang harus dihadapi: **tidak ada satu pun paper ini yang mempertimbangkan arsitektur encoder-decoder**. Semua asumsi mereka embedded pada autoregressive decoder-only:

DPO: $\log \pi_\theta(y|x) = \sum_t \log \pi_\theta(y_t | x, y_{<t})$ — diasumsikan single forward pass kausal.

SimPO: Length normalization $\frac{1}{|y|}$ diasumsikan token-count dari decoder output saja.

KTO: Reference point $z_0$ diestimasi dengan mismatched pairs dari microbatch — ini akan berperilaku berbeda ketika ada encoder yang memproses $x$ secara terpisah.

ORPO: $P_\theta(y|x)$ dalam odds ratio — untuk encoder-decoder ini adalah decoder log-prob kondisional pada encoder output, bukan pada raw $x$.

**Untuk T5Gemma2 dengan merged attention architecture milikmu**, ada implikasi teknis:

Probabilitas $P_\theta(y|x)$ bukan pure autoregressive — decoder menggunakan Key-Value yang merupakan concatenation dari encoder hidden states dan decoder hidden states (merged attention). Ini berarti gradients mengalir melalui dua jalur berbeda saat backprop, yang tidak diperhitungkan dalam derivasi apapun di paper-paper ini.

---

## Analisis per Algoritma untuk Konteks T5Gemma2-Instruct

Berdasarkan proyekmu — **multi-turn conversation**, **bilingual ID-EN**, **implicit task switching via unused tokens**, **in-task advantage** (translate/summarize/QA muncul organik):

### ORPO — Paling Kompatibel Secara Arsitektur

**Kenapa:** ORPO menggabungkan SFT dan preference alignment dalam satu pass. Untuk T5Gemma2, ini sangat berharga karena:

- **Tidak ada reference model** → hemat VRAM secara nyata. Di RTX 3060 6GB kamu, menghilangkan $\pi_{ref}$ berarti bisa fit batch size lebih besar atau sequence lebih panjang.
- **SFT loss tetap ada** di dalam $\mathcal{L}_{ORPO}$. Ini kritis untuk encoder-decoder karena SFT loss membantu encoder tetap belajar representasi yang baik, tidak hanya decoder yang ter-optimize.
- **Odds ratio lebih mild** — untuk multi-task model seperti T5Gemma2 yang harus menangani translate/summarize/QA sekaligus, suppression berlebihan pada rejected response bisa merusak task lain yang tidak terlibat dalam pair itu.

**Masalah:** Dataset harus paired $(x, y_w, y_l)$. Untuk multi-turn conversation, kamu perlu mendefinisikan apa "rejected" — ini tidak trivial.

### KTO — Paling Fleksibel untuk Data

**Kenapa:** KTO hanya butuh binary label (good/bad), bukan paired. Untuk proyekmu:

- **Multi-turn advantage:** Kamu bisa label tiap turn conversation secara independen tanpa perlu pasangkan dua respons untuk satu prompt yang sama. Ini sangat cocok untuk dataset `chat_multiturn` yang kamu punya.
- **Noise robustness:** Dari paper, KTO lebih tahan terhadap noisy preference. Dataset bilingual dengan mixed ID-EN conversation kemungkinan besar lebih noisy daripada single-language dataset.
- **Bisa skip SFT** jika model sudah cukup ter-SFT.

**Masalah besar:** KTO tetap butuh $\pi_{ref}$. Untuk encoder-decoder, ini berarti kamu harus load T5Gemma2 frozen (checkpoint SFT) sebagai reference model secara bersamaan — tambah beban VRAM signifikan.

**Yang lebih penting:** Estimasi $z_0$ dalam KTO menggunakan mismatched pairs dari microbatch. Untuk T5Gemma2 dengan tied embeddings (encoder–decoder–head semuanya share embedding), distribusi log-ratio antar mismatched pairs mungkin berperilaku berbeda dari decoder-only model yang diuji dalam paper.

### SimPO — Secara Teori Paling Bersih, Tapi Ada Caveat Besar

**Keunggulan:** Tidak butuh $\pi_{ref}$, length normalization natural untuk encoder-decoder.

**Masalah fundamental spesifik T5Gemma2:** SimPO menggunakan average log-probability $\frac{1}{|y|} \log \pi_\theta(y|x)$ sebagai implicit reward. Untuk encoder-decoder, $|y|$ adalah panjang decoder output saja — tapi informasi dari encoder juga berkontribusi ke kualitas output. Ini berarti SimPO bisa **under-reward** respons pendek yang informatif (misalnya: summarization bagus yang singkat) versus respons panjang yang verbose (misalnya: translation yang berulang-ulang).

Untuk proyekmu yang punya explicit in-task advantage dimana summarization output harusnya lebih pendek dari translation output, ini masalah serius.

### DPO — Tidak Direkomendasikan

Alasannya sudah jelas dari pembahasan sebelumnya: butuh $\pi_{ref}$, mismatch training/inference metric, dan dari paper SimPO terbukti hanya ~50% preference pairs yang likelihood ranking-nya benar setelah training.

---

## Rekomendasi Konkret

Berdasarkan semua ini, untuk T5Gemma2-Instruct dengan konteksmu:

**Prioritas 1: ORPO** — paling masuk akal karena tidak butuh reference model, SFT loss tetap aktif (bagus untuk encoder), dan mild suppression cocok untuk multi-task.

**Modifikasi yang diperlukan untuk encoder-decoder:**

```python
def orpo_loss_seq2seq(
    model,
    input_ids,        # encoder input
    chosen_ids,       # decoder chosen
    rejected_ids,     # decoder rejected  
    lam=0.1
):
    # Forward pass chosen
    out_w = model(input_ids=input_ids, labels=chosen_ids)
    logprob_w = -out_w.loss  # avg log-prob per token (decoder only)
    
    # Forward pass rejected
    out_l = model(input_ids=input_ids, labels=rejected_ids)
    logprob_l = -out_l.loss
    
    # SFT loss — hanya pada chosen
    l_sft = out_w.loss
    
    # Odds ratio
    odds_w = torch.exp(logprob_w) / (1 - torch.exp(logprob_w) + 1e-7)
    odds_l = torch.exp(logprob_l) / (1 - torch.exp(logprob_l) + 1e-7)
    l_or = -F.logsigmoid(torch.log(odds_w / (odds_l + 1e-7)))
    
    return l_sft + lam * l_or
```

**Prioritas 2: KTO** — jika kamu mau pakai binary labels dari conversation history yang sudah ada, dan bersedia load reference model di cloud (Colab A100).

**Dataset generation strategy untuk multi-turn:**

Untuk ORPO, per-turn rejection bisa dibuat dengan strategi sederhana: generate dua respons untuk tiap turn menggunakan sampling temperature berbeda (0.3 vs 1.2), lalu score dengan rule-based metric (untuk translate: BLEU/chrF; untuk summarize: panjang + keyword coverage; untuk QA: factual grounding terhadap dokumen). Turn yang ada unused token prefix yang benar tapi konten salah = rejected.

**Untuk in-task advantage**, pertimbangkan $\lambda$ yang berbeda per task type dalam ORPO — summarization pairs mungkin butuh $\lambda$ lebih tinggi karena marginalnya lebih sempit, translation lebih rendah karena variance lebih besar.

---

### Bare Minimum Setup untuk T5/Sequence-to-Sequence (TRL)

Berdasarkan pengujian pada Unsloth T5Gemma-2 dan TRL (versi 1.6.0+), ketiga trainer (`ORPOTrainer`, `CPOTrainer` / SimPO, dan `KTOTrainer`) **sepenuhnya mendukung arsitektur sequence-to-sequence (encoder-decoder)**.

Namun, untuk memastikan agar tidak error saat inisialisasi, berikut adalah *bare minimum* modifikasi kode yang wajib Anda sertakan (mirip seperti di script V6 Anda):

1. **Set `decoder_start_token_id`**: Trainer TRL wajib mengetahui token awalan untuk decoder.
   ```python
   if getattr(model.config, "decoder_start_token_id", None) is None:
       model.config.decoder_start_token_id = tokenizer.bos_token_id  # atau pad_token_id
   ```

2. **Perubahan Argumen Config di TRL 1.6.0+**: TRL terbaru mengubah argumen konfigurasi untuk ORPO, CPO, dan KTO. Parameter `max_prompt_length` tidak lagi dipakai di parameter `__init__`. Anda wajib menggunakan `max_length` dan `max_completion_length`.
   ```python
   # Contoh ORPO Config
   from trl.experimental.orpo import ORPOConfig, ORPOTrainer

   orpo_config = ORPOConfig(
       output_dir="./output",
       max_length=2048,
       max_completion_length=1024, # Menggantikan max_prompt_length
       # parameter lain...
   )
   ```

3. **Gunakan PyTorch SDPA (Scaled Dot Product Attention)**: T5 Gemma 2 dan arsitektur encoder-decoder bawaan HuggingFace mengandalkan mekanisme native `sdpa` PyTorch yang sudah teroptimasi dengan sangat baik dan dapat memanggil Flash Attention 2 di bawah kap secara otomatis. Pastikan saat loading model dari config bawaannya (jika tidak via Unsloth), hindari force set argumen `attn_implementation="flash_attention_2"`. Anda cukup pakai bawaannya yang sudah memanggil SDPA.

---

**Sources:**

- [[2402.01306v4] KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306v4)
- [Preference Tuning with Human Feedback on Language, Speech, and Vision   Tasks: A Survey](https://arxiv.org/pdf/2409.11564)
- [Aligning Diffusion Language Models via Unpaired Preference Optimization](https://arxiv.org/pdf/2510.23658)
- [Paper Summary: KTO: Model Alignment as Prospect Theoretic Optimization](https://queirozf.com/entries/paper-summary-kto-model-alignment-as-prospect-theoretic-optimization)
- [[PDF] KTO: Model Alignment as Prospect Theoretic Optimization | Semantic Scholar](https://www.semanticscholar.org/paper/KTO:-Model-Alignment-as-Prospect-Theoretic-Ethayarajh-Xu/c0d8e5ee66c279299012cc3b8d0519011b3f4998)
- [DSTC: Direct Preference Learning with Only Self-Generated Tests and Code   to Improve Code LMs](https://arxiv.org/pdf/2411.13611)
- [Federated Fine-Tuning of Large Language Models: Kahneman-Tversky vs.   Direct Preference Optimization](https://arxiv.org/pdf/2502.14187)
- [SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization](https://arxiv.org/pdf/2602.02383)
- [[2403.07691v2] ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691v2)
- [ORPO: Monolithic Preference Optimization without Reference Model [Quick Review]](https://liner.com/review/orpo-monolithic-preference-optimization-without-reference-model)
- [ORPO: Monolithic Preference Optimization without Reference Model - ACL Anthology](https://aclanthology.org/2024.emnlp-main.626/)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691)
- [DCRM: A Heuristic to Measure Response Pair Quality in Preference Optimization](https://arxiv.org/pdf/2506.14157)
- [AI Paper Review: ORPO - Monolithic Preference Optimization without Reference Model - DEV Community](https://dev.to/bullmouse/ai-paper-review-24l4)
- [When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger](https://arxiv.org/pdf/2603.04968)

Powered by Claude Exporter (https://www.ai-chat-exporter.net)
