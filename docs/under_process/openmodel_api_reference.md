# Dokumentasi Integrasi OpenModel API

Dokumen ini berisi catatan penting terkait konfigurasi integrasi API dengan [OpenModel.ai](https://openmodel.ai), khususnya saat menangani model dari berbagai *provider* (seperti DeepSeek, OpenAI, Anthropic, dll) menggunakan SDK Python atau kerangka *Agentic* seperti `pydantic_ai`.

## Masalah Umum: Error 404 (Route Not Found)

Salah satu masalah yang paling sering ditemui saat mengatur `API_BASE_URL` dan *provider* model adalah mendapatkan *response error* dari OpenModel seperti berikut:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "NOT_FOUND",
    "msg": "route not found"
  }
}
```

Error di atas, yang berformat gaya "OpenAI", terjadi dalam dua skenario utama:
1. **Menggunakan protokol/SDK yang salah untuk model tertentu**.
2. **Duplikasi path `/v1` akibat perilaku otomatis dari Anthropic SDK**.

---

## Aturan Konfigurasi Penting

### 1. Model DeepSeek Wajib Menggunakan Protokol Anthropic
OpenModel melayani model-model seperti OpenAI dan DashScope menggunakan *OpenAI Responses API format*. Namun, untuk model **Anthropic, DeepSeek, Xiaomi, Kimi, MiniMax, dll**, OpenModel mewajibkan penggunaan **Anthropic Messages API format**.

Jika Anda menggunakan model `deepseek-v4-flash` (atau varian DeepSeek lainnya) dengan klien OpenAI, Anda akan selalu mendapatkan error 404 Route Not Found.
- **Pydantic AI**: Saat mendefinisikan agen, Anda **harus** secara eksplisit menggunakan prefix `anthropic:` (contoh: `anthropic:deepseek-v4-flash`).
- **Skrip Kustom**: Pastikan instansiasi klien menggunakan `anthropic.Anthropic()` atau `anthropic.AsyncAnthropic()`, bukan dari *library* `openai`.

### 2. Perilaku Khusus URL pada SDK Anthropic (Resolusi `/v1`)
Terdapat anomali pada cara kerja Anthropic SDK versi Python saat dipadukan dengan `base_url` kustom:
- Jika `base_url` diatur ke `https://api.openmodel.ai`, SDK akan melakukan permintaan ke `https://api.openmodel.ai/messages` (yang mana rute ini tidak ada di server OpenModel dan akan menghasilkan error 404).
- Jika `base_url` diatur ke `https://api.openmodel.ai/v1`, SDK akan melakukan permintaan ke `https://api.openmodel.ai/v1/messages` (rute ini **VALID** dan berhasil).

Oleh karena itu, contoh kode resmi dari OpenModel yang menuliskan `base_url="https://api.openmodel.ai/v1"` adalah konfigurasi yang **benar**.

**Solusi Aman dalam Kode:**
Untuk membuat sistem kebal terhadap apa pun isi di berkas `.env` user, kode yang menggunakan *provider* Anthropic harus dengan pintar memastikan `/v1` ada di ujung string URL. Contoh pengamanan yang sudah diimplementasikan di kode `generate_from_unused_topics.py`:

```python
if provider == "anthropic":
    if API_BASE_URL:
        base = API_BASE_URL
        # Pastikan URL SELALU berakhiran /v1 untuk SDK Anthropic
        if not base.endswith("/v1") and not base.endswith("/v1/"):
            base = base.rstrip("/") + "/v1"
        os.environ["ANTHROPIC_BASE_URL"] = base
```

---

## Ringkasan Konfigurasi `.env` Terbaik

Agar semua kode agen (seperti Pydantic AI) bekerja mulus, konfigurasi `.env` sebaiknya diatur seperti ini:

```env
API_BASE_URL="https://api.openmodel.ai/v1"
API_MODEL="anthropic:deepseek-v4-flash"
API_KEY="sk-om-api-key-anda-disini"
```

Jika kode agen Anda sudah cerdas, ia akan membaca `anthropic:` dari `API_MODEL` dan mem-parsing *provider*-nya dengan benar ke klien Anthropic, sambil memastikan `API_BASE_URL` aman dikonsumsi terlepas dari gaya formatnya.
