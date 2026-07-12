# [MASTER] Analisis Kualitas Dataset Vision & Plan Perbaikan

**Last Updated:** 8 Juli 2026
**Status:** Plan perbaikan siap, script batch fix sudah dibuat & diuji
**Dataset:** `data/multimodal/train_vision.jsonl` (952 records, 250 bermasalah)

---

## Daftar Isi

1. [Temuan Masalah Dataset](#1-temuan-masalah-dataset)
2. [Root Cause di Script Generate](#2-root-cause-di-script-generate)
3. [Impact ke Training](#3-impact-ke-training)
4. [Plan Perbaikan](#4-plan-perbaikan)
5. [Script Batch Fix](#5-script-batch-fix)
6. [Workflow Eksekusi](#6-workflow-eksekusi)
7. [Validasi](#7-validasi)
8. [Post-Fix: Merge & Replace](#8-post-fix-merge--replace)

---
# 1. Temuan Masalah Dataset

Berdasarkan inspeksi `data/multimodal/train_vision.jsonl` (952 records total):

**Statistik Masalah:**
- **250 records bermasalah** (setidaknya 1 turn bermasalah)
- **584 dari 3795 user messages (15.4%)** mengandung `<unused>` token (SEHARUSNYA TIDAK ADA)
- **7 assistant messages** tidak punya `<unused>` prefix
- **702 records sudah bersih** (tidak perlu diubah)

**Contoh Konkret (BAD — User pakai `<unused>`):**

```
ID 300512, [user]: <unused4> Boleh, coba jelaskan bahasa dan budaya ...
ID 300515, [user]: <unused4> Boleh, tolong terjemahkan tulisan yang ...
ID 300503, [user]: <unused6> <unused4> Wah, menarik sekali! Bolehkah ...
ID 300505, [user]: <unused1> <unused3> Terima kasih atas informasinya ...
```

**Contoh Konkret (BAD — Assistant tanpa `<unused>`):**

```
ID XXX, [assistant]: Di gambar halaman 2 terdapat teks tentang ...
  (seharusnya: <unused4> Di gambar halaman 2 terdapat teks tentang ...)
```

**Pola Masalah:**
1. LLM menulis `<unusedX>` di awal pesan **USER** (harusnya hanya di assistant)
2. LLM kadang **lupa** menulis `<unusedX>` di awal pesan **ASSISTANT**
3. Kualitas grounding visual tidak konsisten (beberapa jawaban terlalu umum)
4. Beberapa percakapan dipaksa 10 turn sehingga tidak natural

---

# 2. Root Cause di Script Generate

**File:** `scripts/dataset/generate_multimodal_conv.py`

**Mekanisme Script:**
1. pydantic-ai Agent diberi gambar + system prompt
2. Agent menggunakan tool `append_turn_pair(human_user_message, task_prefixes, ai_assistant_message)` untuk build percakapan
3. Agent pakai `edit_turn_content` untuk koreksi
4. Agent pakai `get_conversation_status` untuk self-review
5. Output: `ConversationResult` (is_valid, rationale)

**BUG 1 — Tool `append_turn_pair` tidak validasi user message (line 300-330):**

```python
# Tool HANYA append prefix ke assistant, BENAR:
final_assistant_message = f"{prefix_str} {ai_assistant_message}"
state.turns.append({"role": "user", "content": human_user_message})
state.turns.append({"role": "assistant", "content": final_assistant_message})
```

Tool menerima `human_user_message` apa adanya. **LLM sering menulis `<unused4>` di dalam `human_user_message`** sebagai bagian natural teks user — tool tidak menolaknya.

**BUG 2 — System prompt tidak cukup tegas (line 280):**

```
"5. PENTING: `human_user_message` HARUS berisi ucapan PENGGUNA MANUSIA.
   `ai_assistant_message` HARUS berisi respons dari BOT AI. JANGAN PERNAH TERTUKAR!"
```

Instruksi ada tapi LLM (google/gemma-4-31b-it:free) tetap sering menulis prefix di user message.

**BUG 3 — Tidak ada post-processing (line 651-668):**

Setelah generation, script langsung tulis ke file. Tidak ada filter yang strip `<unused>` dari user messages.

**BUG 4 — Self-review tidak efektif:**

Tool `get_conversation_status` dan `edit_turn_content` ada, tapi LLM tidak konsisten menggunakannya untuk strip prefix dari user.

**Kesimpulan:** Root cause = LLM tidak patuh instruksi + tool tidak punya guardrail + tidak ada post-processing cleanup.

---

# 3. Impact ke Training

**Training text-only v6 (SFT + ORPO) sudah selesai** dengan logit masking:
- SUPPRESS_VISION = `[255999, 256000, 256001]` (3 image token di-suppress)
- Model tidak boleh generate token image

**Impact jika dataset vision pakai data bermasalah:**

1. **User dengan `<unused>` prefix** → Model belajar bahwa user boleh pakai token task-prefix
   → Ngelanggar desain (prefix HANYA untuk assistant sebagai control signal)
   → Saat inferensi, user prompt normal tidak akan punya prefix → distribusi training≠inferensi

2. **Assistant tanpa prefix** → Model tidak belajar mapping task→prefix
   → Control signal (summarize/translate/qa/etc) jadi inconsistent

3. **Grounding lemah** → Model tidak belajar merujuk gambar dengan benar
   → Saat vision training, behavior buruk terbawa

4. **250/952 records (26%) bermasalah** → Cukup besar untuk merusak kualitas akhir

**Rekomendasi:** FIX dataset sebelum vision training (test_v6_vision_unsloth.py / working-molab-v6-vision-unsloth.py).

---

# 4. Plan Perbaikan

**Strategi:** Perbaiki SEMUA 250 records bermasalah via batch processing
(dengan agent antigravity yang bisa lihat gambar), output ke file BARU
(`train_vision_fixed.jsonl`) agar original tetap aman sebagai backup.

**2 Pendekatan:**

### A. Manual Re-Write (DIPAKAI — karena butuh lihat gambar)
- Pakai script batch (generate + commit) seperti workflow ORPO
- Agent antigravity lihat gambar + percakapan lama, tulis ulang high-quality
- Lebih lambat tapi kualitas terjamin (grounded + natural)

### B. Automated Strip (TIDAK CUKUP — hanya cosmetic)
- Bisa strip `<unused>` dari user messages dengan regex
- TAPI tidak memperbaiki grounding / naturalness / assistant tanpa prefix
- Hanya sebagai fallback jika manual terlalu lambat

**Aturan Ketat untuk Re-Write:**
1. User messages: HAPUS semua `<unusedX>`, bahasa natural murni
2. Assistant: WAJIB `<unused1>`-`<unused6>` di awal (max 3 token)
3. Visually grounded: jawaban merujuk detail gambar
4. Natural: 6/8/10 turn (jangan dipaksa)
5. 📷 token tetap di awal user turn pertama (jumlah = jumlah gambar)
6. System prompt tetap sama

---

# 5. Script Batch Fix

**File 1: `scratch/generate_batch_fix_vision.py`**
- Load `train_vision.jsonl`, filter records bermasalah (`has_issues()`)
- Batch 10 conversations (5 doc + 5 random) per run
- Copy gambar ke folder antigravity, generate markdown input
- Highlight masalah (user `<unused>` = merah, assistant tanpa prefix = warning)
- Output: `vision_fix_batch_input.md` di folder antigravity
- Progress: `scratch/fix_vision_progress.json`

**File 2: `scratch/commit_fix_vision_batch.py`**
- Baca `scratch/temp_batch_fix_out.json` (hasil agent)
- Validasi: user tanpa `<unused>`, assistant wajib prefix, 📷 di user pertama
- Append ke `data/multimodal/train_vision_fixed.jsonl` (file BARU)
- Update progress
- Cleanup temp file

**Status:** ✅ Script sudah dibuat & diuji (Batch 1: 7 conversations)

---

# 6. Workflow Eksekusi

```
LOOP sampai semua 250 records selesai:
  1. python scratch/generate_batch_fix_vision.py
     → Output: vision_fix_batch_input.md (folder antigravity)
  2. Agent antigravity baca markdown, lihat gambar
     → Tulis ulang percakapan high-quality
     → Output: scratch/temp_batch_fix_out.json
  3. python scratch/commit_fix_vision_batch.py
     → Validasi + append ke train_vision_fixed.jsonl
     → Update progress
```

**Progress tracking:** `scratch/fix_vision_progress.json`
- `{ "processed_ids": [300072, 300512, ...] }`
- Resume otomatis (skip IDs sudah diproses)

---

# 7. Validasi

**Di `commit_fix_vision_batch.py` (`validate_record`):**

| Check | Condition | Action |
|-------|-----------|--------|
| User ada `<unused>` | `re.search(r"<unused\d+>", user_content)` | ❌ REJECT |
| Assistant tanpa prefix | `not re.search(r"<unused\d+>", asst_content)` | ❌ REJECT |
| Assistant >3 prefix | `len(prefixes) > 3` | ❌ REJECT |
| User pertama tanpa 📷 | `"📷" not in user_msgs[0]` | ❌ REJECT |
| Ada system/user/assistant | roles check | ❌ REJECT |

**Output:**
- Valid records → append ke `train_vision_fixed.jsonl`
- Invalid records → print error, skip (tidak di-commit)
- Progress di-update hanya untuk valid records

---

# 8. Post-Fix: Merge & Replace

**Setelah semua 250 records fixed:**

1. Load `train_vision.jsonl` (original 952)
2. Load `train_vision_fixed.jsonl` (250 fixed)
3. Build dict: `{id: record}` dari fixed
4. Iterate original:
   - Jika id ada di fixed → pakai versi fixed
   - Jika tidak → pakai original (sudah bersih)
5. Write ke `train_vision.jsonl` (replace) ATAU `train_vision_v2.jsonl`

**Backup:** Original tetap ada sebagai `train_vision_original_backup.jsonl`

**Catatan ORPO:**
- 200 records dari train_vision digunakan sebagai basis ORPO
- Jika salah satu dari 200 itu termasuk 250 bermasalah → juga perlu di-fix
- Cek `orpo_progress.json` (200 IDs) vs `fix_vision_progress.json`
- Re-generate ORPO batch untuk IDs yang overlap (jika perlu)

**Status:** ⏳ BELUM DIKERJAKAN (menunggu agent antigravity selesaikan batch fix)
