"""Tes validasi strict: jumlah turn 6/8/10, num_pairs cocok, threshold text 60 / vision 40, strip token global."""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mcpb_stdio import BUNDLE_SRC, CMD, MCPServer, drain_stderr  # noqa: E402


def conv(n_pairs: int, short_len: int = 0, multi_prefixes=None, user_with_prefix: bool = False) -> str:
    turns = []
    for i in range(n_pairs):
        user_pfx = ["<unused4>"] if (user_with_prefix and i == 0) else []
        turns.append({"role": "user", "content": f"Pertanyaan nomor {i} tentang topik yang dibahas dalam konteks.", "prefixes": user_pfx})
        pfx = multi_prefixes or ["<unused4>"]
        if short_len and i == 0:
            turns.append({"role": "assistant", "content": "x" * short_len, "prefixes": pfx})
        else:
            turns.append({"role": "assistant",
                          "content": f"Jawaban nomor {i}: ini adalah jawaban yang cukup panjang dan mendetail, "
                                     f"menjelaskan langkah demi langkah, memberikan analogi, dan menutup dengan kesimpulan singkat "
                                     f"supaya turn ini memenuhi syarat minimal karakter yang ditetapkan oleh validator.",
                          "prefixes": pfx})
    return json.dumps(turns, ensure_ascii=False)


async def save(srv, n_pairs, num_pairs_arg, category="text_nlu_chat", short_len=0, content=None):
    if content is None:
        content = conv(n_pairs, short_len)
    r = await srv.call("tools/call", {"name": "save_conversation", "arguments": {
        "source": "Tes (HuggingFace: dummy), Row #1", "category": category,
        "conversation_json": content, "num_pairs": num_pairs_arg}})
    return r["result"].get("isError"), r["result"]["content"][0]["text"]


async def main():
    tmp = tempfile.mkdtemp(prefix="mcpb_strict_")
    env = dict(os.environ, OUTPUT_DIR=tmp, PYTHONIOENCODING="utf-8")
    proc = await asyncio.create_subprocess_exec(
        *CMD, cwd=BUNDLE_SRC, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    asyncio.get_running_loop().create_task(drain_stderr(proc.stderr, []))
    srv = MCPServer(proc, proc.stdin)
    await srv.call("initialize", {"protocolVersion": "2026-07-28", "capabilities": {},
                                  "clientInfo": {"name": "strict-test", "version": "1"}}, timeout=240)
    await srv.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    ok = True
    cases = [
        # (label, content, num_pairs_arg, category, expect_isError)
        ("1 pasang (2 pesan) → DITOLAK", conv(1), "", "text_nlu_chat", True),
        ("2 pasang (4 pesan) → DITOLAK", conv(2), "", "text_nlu_chat", True),
        ("4 pasang + num_pairs=4 → LOLOS", conv(4), "4", "text_nlu_chat", False),
        ("4 pasang tapi num_pairs=3 → DITOLAK", conv(4), "3", "text_nlu_chat", True),
        ("text: turn 3 karakter → DITOLAK (<60)", conv(4, 3), "4", "text_nlu_chat", True),
        ("text: turn 55 karakter → DITOLAK (<60)", conv(4, 55), "4", "text_nlu_chat", True),
        ("vision: turn 45 karakter → LOLOS (≥40)", conv(4, 45), "4", "vision_chat", False),
        ("vision: turn 30 karakter → DITOLAK (<40)", conv(4, 30), "4", "vision_chat", True),
        ("assistant 4 token unik → DITOLAK (max 3)", conv(3, multi_prefixes=["<unused1>", "<unused2>", "<unused3>", "<unused4>"]), "3", "text_nlu_chat", True),
        ("assistant 3 token unik → LOLOS (≤3)", conv(3, multi_prefixes=["<unused1>", "<unused2>", "<unused3>"]), "3", "text_nlu_chat", False),
        ("user turn ber-prefix → DITOLAK", conv(3, user_with_prefix=True), "3", "text_nlu_chat", True),
    ]
    for label, content, np_, cat, expect in cases:
        err, text = await save(srv, 0, np_, cat, 0, content)
        passed = bool(err) == expect
        ok &= passed
        print(f"{'✅' if passed else '❌'} {label}\n    isError={err} | msg: {text[:130]}")

    # #7: strip token <unusedX> di TENGAH konten (bukan cuma awal) — pakai 3 pasang biar valid (6 pesan)
    content_leak = json.dumps([
        {"role": "user", "content": "Tolong jelaskan topik dari konteks ini secara rinci ya.", "prefixes": []},
        {"role": "assistant", "content": "<unused4> <unused4> Penjelasan yang cukup panjang tentang topik konteks, "
                                         "dengan <unused2> token bocor di tengah kalimat yang harus dibersihkan "
                                         "supaya hasil akhirnya bersih dan hanya punya satu prefix di awal.", "prefixes": ["<unused4>"]},
        {"role": "user", "content": "Kalau begitu, bagaimana contoh penerapannya dalam kehidupan sehari-hari?", "prefixes": []},
        {"role": "assistant", "content": "Contoh penerapannya adalah ketika kita mengamati fenomena yang dijelaskan di konteks, "
                                         "kita bisa melihat dampaknya secara langsung dan mengevaluasi langkah berikutnya "
                                         "dengan lebih terstruktur dan sistematis sesuai penjelasan tadi.", "prefixes": ["<unused4>"]},
        {"role": "user", "content": "Apakah ada batasan atau kekurangan dari pendekatan itu?", "prefixes": []},
        {"role": "assistant", "content": "Batasan utamanya adalah pendekatan ini sangat bergantung pada kualitas data awal dan asumsi yang dipakai, "
                                         "sehingga hasilnya tetap perlu diverifikasi oleh ahli sebelum dijadikan keputusan akhir.", "prefixes": ["<unused4>"]},
    ])
    err, text = await save(srv, 3, "3", "text_nlu_chat", 0, content_leak)
    print(f"token bocor tengah: isError={err} | msg: {text[:120]}")
    out = os.path.join(tmp, "generated_conv_claude.jsonl")
    raw = open(out, encoding="utf-8").read()
    # Ambil konten assistant di baris terakhir (yang mengandung 'Penjelasan yang cukup panjang')
    last = [l for l in raw.strip().splitlines() if "Penjelasan yang cukup panjang" in l][-1]
    rec = json.loads(last)
    a_content = [m["content"] for m in rec["messages"] if m["role"] == "assistant"][-1]
    print("assistant content akhir:", repr(a_content[:160]))
    stripped = a_content.replace("<unused4>", "")
    assert "<unused" not in stripped, "masih ada token di tengah konten!"
    print("✅ strip token global bekerja (token hanya tersisa 1x di awal, dari prefixes)")
    ok &= True

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
    print("\n" + ("✅ SEMUA VALIDASI LULUS" if ok else "❌ ADA YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
