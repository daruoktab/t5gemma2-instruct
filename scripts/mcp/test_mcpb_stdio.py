"""Uji JSON-RPC stdio untuk MCPB generate-conv-indonesia v2.1.8 (random-access).

Menjalankan server via `uv run --with ...` (env ephemeral, mcp>=2.0.0) lalu
berbicara protokol MCP stdio: initialize → tools/list → list_sources →
sample_row (rows / csv / cvqa vision) → read_image → save_conversation →
get_output_stats → verifikasi exclusion (seed sama tidak boleh row sama).
"""
import asyncio
import json
import os
import sys
import tempfile

BUNDLE_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "mcpb", "generate-conv-indonesia", "src"))

DEP_WITHS = ["mcp>=2.0.0", "datasets>=5.0.0", "pillow", "pyarrow", "fsspec",
             "huggingface_hub", "python-dotenv", "httpx"]

CMD = ["uv", "run", "--no-project", "--python", "3.12"]
for w in DEP_WITHS:
    CMD += ["--with", w]
CMD += ["python", "-u", "server.py"]


def shorten(s, n=300):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"…({len(s)} chr)"


class MCPServer:
    def __init__(self, proc, writer):
        self.proc, self.writer = proc, writer
        self._id = 0
        self._buf = b""

    async def send(self, obj):
        self.writer.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        await self.writer.drain()

    async def _readline(self, timeout):
        # Line reader manual (StreamReader.readline limit 64KB terlalu kecil untuk ImageContent)
        while b"\n" not in self._buf:
            chunk = await asyncio.wait_for(self.proc.stdout.read(65536), timeout=timeout)
            if not chunk:
                if not self._buf:
                    return None  # EOF
                break
            self._buf += chunk
        idx = self._buf.find(b"\n")
        if idx == -1:
            line, self._buf = self._buf, b""
        else:
            line, self._buf = self._buf[: idx + 1], self._buf[idx + 1:]
        return line

    async def recv(self, timeout):
        line = await self._readline(timeout)
        if line is None:
            raise RuntimeError("server stdout EOF")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[raw line] {line[:200]!r}", file=sys.stderr)
            raise

    async def call(self, method, params=None, timeout=180):
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        await self.send(msg)
        while True:
            resp = await self.recv(timeout)
            if resp.get("id") == self._id:
                return resp
            # notifikasi (logs) — lewati
            print(f"  [notif] {resp.get('method', resp)}", file=sys.stderr)


async def drain_stderr(stream, sink):
    while True:
        line = await stream.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").rstrip()
        print(f"  [server] {line}", file=sys.stderr)
        sink.append(line)


async def main():
    tmp = tempfile.mkdtemp(prefix="mcpb_test_")
    print(f"OUTPUT_DIR = {tmp}")
    env = dict(os.environ, OUTPUT_DIR=tmp, PYTHONIOENCODING="utf-8")
    proc = await asyncio.create_subprocess_exec(
        *CMD, cwd=BUNDLE_SRC, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    stderr_lines = []
    asyncio.get_running_loop().create_task(drain_stderr(proc.stderr, stderr_lines))
    srv = MCPServer(proc, proc.stdin)

    try:
        r = await srv.call("initialize", {
            "protocolVersion": "2026-07-28",
            "capabilities": {},
            "clientInfo": {"name": "test-stdio", "version": "1.0"},
        }, timeout=240)
        print("initialize:", json.dumps(r.get("result", r), ensure_ascii=False)[:200])
        await srv.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        r = await srv.call("tools/list")
        names = [t["name"] for t in r["result"]["tools"]]
        print("tools:", names)
        assert "get_progress" in names, "get_progress harus ada"

        r = await srv.call("tools/call", {"name": "list_sources", "arguments": {"category": "all"}})
        print("list_sources ->", shorten(r["result"]["content"][0]["text"]))

        # 0) get_progress awal (kosong) — next harus text (rasio 0/0)
        r = await srv.call("tools/call", {"name": "get_progress", "arguments": {}})
        prog0 = json.loads(r["result"]["content"][0]["text"])
        print("get_progress awal ->", shorten(prog0))
        assert "prefix_stats" in prog0 and "prefix_hint" in prog0, "prefix_stats/hint harus ada"
        assert prog0["prefix_stats"] == {"<unused1>": 0, "<unused2>": 0, "<unused3>": 0, "<unused4>": 0, "<unused5>": 0, "<unused6>": 0}
        assert "pairs_stats" in prog0 and "pairs_hint" in prog0, "pairs_stats/hint harus ada"
        assert prog0["pairs_stats"] == {"3": 0, "4": 0, "5": 0}

        # 0b) sample_row AUTO (tanpa category) — harus text (karena rasio text paling tertinggal)
        r = await srv.call("tools/call", {"name": "sample_row", "arguments": {}})
        auto = json.loads(r["result"]["content"][0]["text"])
        print("sample AUTO ->", auto["category"], "| auto_category:", auto.get("auto_category"), "|", auto["source"])
        assert "prefix_hint" in auto and "prefix_stats" in auto, "sample_row harus punya prefix hint/stats"
        assert "pairs_hint" in auto and "pairs_stats" in auto, "sample_row harus punya pairs hint/stats"

        # 1) rows access (text) — seed tetap
        r = await srv.call("tools/call", {"name": "sample_row",
                                          "arguments": {"category": "text", "source_key": "IndoCareer", "seed": 7}})
        row1 = json.loads(r["result"]["content"][0]["text"])
        print("sample IndoCareer (seed=7) ->", row1["source"], "| ctx:", shorten(row1["raw_context"], 120))

        # 2) vision KTP (rows) — seed=514; hasilnya HARUS ada ImageContent langsung
        r = await srv.call("tools/call", {"name": "sample_row",
                                          "arguments": {"category": "vision", "source_key": "KTP-VLM", "seed": 514}})
        ktp = json.loads(r["result"]["content"][0]["text"])
        print("sample KTP-VLM (seed=514) ->", ktp["source"], "| image_ref:", shorten(ktp["image_ref"], 100),
              "| img_size:", ktp["image_size"], "| etika:", shorten(ktp.get("etika", ""), 90))
        inline = r["result"]["content"][1] if len(r["result"]["content"]) > 1 else None
        print("inline image part ->", inline.get("type") if inline else None,
              "| mime:", inline.get("mime_type") if inline else None,
              "| b64 len:", len(inline.get("data", "")) if inline else 0)
        assert inline and inline.get("type") == "image" and inline.get("data"), "KTP vision harus inline image"
        assert len(inline["data"]) < 1000 * 1024, f"inline image melebihi 1MB: {len(inline['data'])} chars"
        print("✅ inline image < 1MB (kompresi on-the-fly bekerja)")

        # 3) read_image tetap jalan (fallback)
        r = await srv.call("tools/call", {"name": "read_image", "arguments": {"image_ref": ktp["image_ref"]}}, timeout=180)
        content = r["result"]["content"][0]
        print("read_image -> type:", content.get("type"), "| b64 len:", len(content.get("data", "")))

        # 4) CSV access (IndoMMLU — unduh sekali ke cache)
        r = await srv.call("tools/call", {"name": "sample_row",
                                          "arguments": {"category": "text", "source_key": "IndoMMLU", "seed": 3}}, timeout=300)
        mmlu = json.loads(r["result"]["content"][0]["text"])
        print("sample IndoMMLU (csv) ->", mmlu["source"], "| note:", mmlu["note"])

        # 5) CVQA vision (filter → fallback berlapis) — bisa lama pertama kali
        r = await srv.call("tools/call", {"name": "sample_row",
                                          "arguments": {"category": "vision", "source_key": "CVQA-Indonesia"}}, timeout=420)
        raw_cvqa = r["result"]["content"][0]["text"]
        print("CVQA raw (isError={}):".format(r.get("isError")), shorten(raw_cvqa, 400))
        cvqa = json.loads(raw_cvqa)
        print("sample CVQA ->", cvqa["source"], "| image_available:", cvqa["image_available"],
              "| note:", cvqa["note"], "| ctx:", shorten(cvqa["raw_context"], 150))

        # 6) simpan percakapan (source PERSIS dari sample_row — untuk exclude); 3 pasang = 6 pesan (aturan strict)
        conv = json.dumps([
            {"role": "user", "content": "Apa itu fotosintesis dan mengapa penting bagi tumbuhan?", "prefixes": []},
            {"role": "assistant", "content": "Fotosintesis adalah proses tumbuhan mengubah cahaya matahari menjadi energi kimia. Proses ini penting karena menjadi sumber makanan utama bagi tumbuhan sekaligus penghasil oksigen yang kita hirup setiap hari.", "prefixes": ["<unused4>"]},
            {"role": "user", "content": "Kalau begitu, bagian tumbuhan mana yang berperan paling besar dalam proses ini?", "prefixes": []},
            {"role": "assistant", "content": "Daun adalah bagian yang paling berperan karena mengandung klorofil, pigmen hijau yang menangkap energi cahaya. Klorofil inilah yang membuat fotosintesis bisa berlangsung secara efisien di siang hari.", "prefixes": ["<unused4>", "<unused1>"]},
            {"role": "user", "content": "Apakah ada faktor lingkungan yang bisa mengganggu proses fotosintesis?", "prefixes": []},
            {"role": "assistant", "content": "Ada beberapa faktor, seperti kekurangan air, suhu yang terlalu panas atau dingin, serta minimnya cahaya matahari. Ketiganya bisa menurunkan laju fotosintesis sehingga pertumbuhan tumbuhan ikut terhambat.", "prefixes": ["<unused4>"]},
        ])
        r = await srv.call("tools/call", {"name": "save_conversation", "arguments": {
            "source": row1["source"], "category": "text_nlu_chat",
            "conversation_json": conv, "num_pairs": "3"}})
        print("save_conversation ->", r["result"]["content"][0]["text"])

        # 7) stats
        out = os.path.join(tmp, "generated_conv_agent.jsonl")
        r = await srv.call("tools/call", {"name": "get_output_stats", "arguments": {"output_path": out}})
        print("get_output_stats ->", shorten(r["result"]["content"][0]["text"]))

        # 7b) get_progress setelah 1 text disimpan → text.done=1, next harus vision, prefix_stats terisi
        r = await srv.call("tools/call", {"name": "get_progress", "arguments": {}})
        prog1 = json.loads(r["result"]["content"][0]["text"])
        print("get_progress setelah save -> text:", prog1["text"], "| vision:", prog1["vision"],
              "| next:", prog1["next_category"], "| next_src:", prog1["next_source_key"],
              "| prefix_stats:", prog1["prefix_stats"], "| hint:", shorten(prog1["prefix_hint"], 80),
              "| pairs_stats:", prog1["pairs_stats"], "| pairs_hint:", shorten(prog1["pairs_hint"], 80))
        assert prog1["text"]["done"] == 1 and prog1["next_category"] == "vision", "kuota/next salah"
        assert sum(prog1["prefix_stats"].values()) >= 1, "prefix_stats harus terisi setelah save"
        assert prog1["pairs_stats"].get("3", 0) == 1, "pairs_stats harus terisi setelah save (num_pairs=3)"

        # 8) exclusion: seed sama 7 → offset pertama pasti row yg sama → harus diskip
        r = await srv.call("tools/call", {"name": "sample_row",
                                          "arguments": {"category": "text", "source_key": "IndoCareer", "seed": 7}})
        row2 = json.loads(r["result"]["content"][0]["text"])
        same = row1["source"] == row2["source"]
        print("re-sample IndoCareer (seed=7) ->", row2["source"], "| SAME row (exclude GAGAL):", same)
        if same:
            print("❌ EXCLUDE TIDAK BEKERJA")
        else:
            print("✅ EXCLUDE BEKERJA (row yang disimpan tidak di-sample ulang)")

        print("\n=== SEMUA TEST SELESAI ===")
    finally:
        rc = proc.returncode
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
        print(f"\n--- server exit rc={rc} ---")
        print("--- stderr server (tail) ---")
        for line in stderr_lines[-25:]:
            print(" ", line)


if __name__ == "__main__":
    asyncio.run(main())
