# 🔬 Deep-Dive 01 — Token-Level Policy Optimization & Alignment Mechanisms (V8 Roadmap)

**Tanggal Analysis:** 6 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `TLPO` (2604.26553): [pdfs/TLPO_2604.26553.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/TLPO_2604.26553.pdf) (`pages/TLPO/p01.png` s.d. `p21.png`)
- `FT_OBJECTIVES_SAFETY` (2601.12639): [pdfs/FT_OBJECTIVES_SAFETY_2601.12639.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/FT_OBJECTIVES_SAFETY_2601.12639.pdf) (`pages/FT_OBJECTIVES_SAFETY/p01.png` s.d. `p17.png`)
- `ALIGNMENT_MECHANISTIC` (2606.09850): [pdfs/ALIGNMENT_MECHANISTIC_2606.09850.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/ALIGNMENT_MECHANISTIC_2606.09850.pdf) (`pages/ALIGNMENT_MECHANISTIC/p01.png` s.d. `p30.png`)
- `CULTURE_ID` (2607.21016): [pdfs/CULTURE_ID_2607.21016.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/CULTURE_ID_2607.21016.pdf) (`pages/CULTURE_ID/p01.png` s.d. `p24.png`)

---

## 1. TLPO: Token-Level Policy Optimization (Samsung SDS, Apr 2026)
**arXiv:** [2604.26553](https://arxiv.org/abs/2604.26553)

### A. Masalah & Motivasi Utama
Dalam fine-tuning preferensi sekuens (sequence-level alignment) seperti DPO, GRPO, atau ORPO, penalti diterapkan pada *keseluruhan sekuens respons* sebagai satu kesatuan (*monolithic unit*). Hal ini menimbulkan **catastrophic forgetting** dan **language confusion** (pencampuran kata/bahasa asing dalam respons), karena model menekan token-token pendukung yang valid di sekitar token yang salah bahasa. TLPO memecahkan masalah ini dengan melokalisasi pembaruan kebijakan (*policy updates*) khusus di posisi token di mana kesalahan bahasa terjadi (*confusion point* $c$).

### B. Formulasi Matematika & Algoritma
1. **Deteksi Confusion Point ($c$):**
   Mencari indeks token pertama $y_c$ dalam sekuens respons $y$ yang melanggar himpunan karakter bahasa target (misal: Bahasa Indonesia).
2. **Probability-Ranked Token Exploration:**
   Di titik $c$, ambil top-$N$ candidate tokens $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$ berdasarkan probabilitas $\pi_\theta(\cdot \mid x, y_{<c})$.
3. **Fungsi Reward & Advantage Terbobot Probabilitas:**
   $$\mu = \frac{\sum_{j=1}^N \pi_{\theta_{\text{old}}}(t_j \mid x, y_{<c}) R(t_j)}{\sum_{j=1}^N \pi_{\theta_{\text{old}}}(t_j \mid x, y_{<c})}$$
   $$A_i = \frac{1}{Z} \cdot \pi_{\theta_{\text{old}}}(t_i \mid x, y_{<c}) \left( R(t_i) - \mu \right)$$
   di mana $Z = \sum_{j=1}^N \pi_{\theta_{\text{old}}}(t_j \mid x, y_{<c}) |R(t_j) - \mu|$ adalah konstanta normalisasi agar $\sum |A_i| = 1$.
4. **Fungsi Loss TLPO:**
   $$\mathcal{J}_{\text{TLPO}}(\theta) = \mathbb{E}\left[ \frac{1}{N} \sum_{t_i \in \mathcal{T}} \min\left( \frac{\pi_\theta(t_i \mid x, y_{<c})}{\pi_{\theta_{\text{old}}}(t_i \mid x, y_{<c})} A_i, \text{clip}\left(\frac{\pi_\theta(t_i \mid x, y_{<c})}{\pi_{\theta_{\text{old}}}(t_i \mid x, y_{<c})}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

### C. Temuan Kunci & Data Empiris
- On **Gemma-3-4B-IT**, SFT mengalami penurunan akurasi penalaran umum (GSM8K/MMLU) dari 58.35% ke 50.71%. DPO dan ORPO turun ke 55.94% dan 55.12%.
- **TLPO mempertahankan akurasi general pada 58.08%** (hampir sama dengan baseline) sembari meningkatkan *Response Pass Rate (RPR)* kepatuhan bahasa target dari 96.68% ke **99.19%**.

---

## 2. Objective Matters: Fine-Tuning Objectives Shape Safety & Robustness (Jan 2026)
**arXiv:** [2601.12639](https://arxiv.org/abs/2601.12639)

### A. Masalah & Motivasi Utama
Fine-tuning pada data instruksi yang tampaknya aman (benign SFT) dapat merusak ketahanan terhadap *jailbreak attacks* dan menginduksi *persona drift* (pergeseran sifat model).

### B. Temuan Kunci
- Pada budget pelatihan kecil (<100k token), pilihan objective tidak berpengaruh besar pada keamanan.
- Pada budget besar (>400k token), **SFT dan DPO mengalami lonjakan titik lemah kerentanan (ASR naik drastis)**.
- **ORPO dan KL-regularization** bertindak sebagai peredam *persona drift* dan mempertahankan ketahanan lawan (*adversarial robustness*).

---

## 3. Mechanistic Analysis of Alignment Algorithms in LLMs (Mei 2026)
**arXiv:** [2606.09850](https://arxiv.org/abs/2606.09850)

### A. Temuan Mekanistik via SAE & Linear Probing
- KTO dan GRPO meningkatkan pemisahan linier preferensi (*linear separability*) melalui *sparse recruitment of high-salience features*.
- DPO dan ORPO menyebabkan rotasi non-konstruktif dan *feature attenuation* (pelemahan sinyal fitur).
- Sinyal preferensi terpusat terutama pada **middle-to-late layers** (layer 12 s.d. 24 pada model 28-layer).

---

## 4. CultureTalk-ID: Dialogue Benchmark for Cultural Commonsense Indonesia (MBZUAI, Jul 2026)
**arXiv:** [2607.21016](https://arxiv.org/abs/2607.21016)

### A. Karakteristik Benchmark
- 4.496 dialog terkurasi manusia meliputi 10 provinsi Indonesia, 11 bahasa daerah (Jawa, Sunda, Minang, Bali, Bugis, Wamesa, Uab Meto, dll.), dan 13 topik budaya.
- 3 Task Utama: Dialogue MCQ Cultural Commonsense, Machine Translation (ID $\leftrightarrow$ Local), dan Language Steering.

---

## 🛠️ Rencana Implementasi Presisi pada Kode Pipeline V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Penambahan Loss TLPO pada Multitask Trainer V8
Di `working-molab-v7-combined-unsloth.py`, buat fungsi pembantu `compute_tlpo_loss()` dan integrasikan ke dalam `custom_loss_func`:

```python
import torch
import torch.nn.functional as F

def detect_language_confusion_points(input_ids, logits, target_lang_vocab_set):
    """
    Menemukan indeks token pertama di mana token tidak masuk dalam target_lang_vocab_set.
    """
    preds = torch.argmax(logits, dim=-1)
    # Mask untuk token yang berada di luar kosa kata bahasa target (misal Indonesia)
    confusion_mask = ~torch.isin(preds, target_lang_vocab_set)
    return confusion_mask

def compute_tlpo_loss(logits, labels, old_logits, target_lang_vocab_set, beta=0.1, clip_eps=0.2):
    """
    Formulasi Token-Level Policy Optimization (TLPO) untuk menekan language confusion.
    """
    confusion_mask = detect_language_confusion_points(labels, logits, target_lang_vocab_set)
    if not confusion_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    probs = F.softmax(logits, dim=-1)
    old_probs = F.softmax(old_logits.detach(), dim=-1)
    
    # Ambil top-N candidate tokens (N=16) pada confusion point
    top_n_probs, top_n_indices = torch.topk(old_probs, k=16, dim=-1)
    
    # Hitung reward: +1 jika token tergolong bahasa target, -1 jika confusion
    rewards = torch.isin(top_n_indices, target_lang_vocab_set).float() * 2.0 - 1.0
    
    # Advantage terbobot probabilitas (Eq 5 & 6 TLPO)
    weighted_mean_reward = torch.sum(top_n_probs * rewards, dim=-1, keepdim=True) / (torch.sum(top_n_probs, dim=-1, keepdim=True) + 1e-8)
    advantages = top_n_probs * (rewards - weighted_mean_reward)
    advantages = advantages / (torch.sum(torch.abs(advantages), dim=-1, keepdim=True) + 1e-8)
    
    # Ratio PPO-style
    curr_top_n_probs = torch.gather(probs, -1, top_n_indices)
    ratio = curr_top_n_probs / (top_n_probs + 1e-8)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Penalty KL Divergence
    kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), old_probs, reduction='batchmean')
    
    return policy_loss + beta * kl_loss
```

### 2. Penargetan Layer LoRA Presisi (Mechanistic Alignment)
Di bagian `FastLanguageModel.get_peft_model`:
```python
# Berdasarkan temuan Mechanistic Analysis (2606.09850), fokuskan LoRA pada layer 12 s.d. 24
target_layers = [f"decoder.layers.{i}" for i in range(12, 24)]
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=list(range(12, 24)), # membatasi pergeseran representasi hanya pada layer aliansi
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
)
```

### 3. Integrasi Evaluasi CultureTalk-ID pada Pipeline Validation
Tambahkan modul evaluasi `scripts/eval/evaluate_culturetalk_id.py` yang memuat dataset `CultureTalk-ID` untuk menguji:
- **Dialogue MCQ Accuracy** (Mengukur RPR pada dialog 11 bahasa daerah).
- **Language Steering BLEU** (Mengukur kemampuan beralih antara Bahasa Indonesia dan bahasa daerah tanpa pencampuran yang salah).
