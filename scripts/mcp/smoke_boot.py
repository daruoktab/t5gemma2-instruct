"""Smoke test cepat: boot server, cek katalog 15 sumber, IndoRad-test hilang, train jalan."""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mcpb_stdio import BUNDLE_SRC, CMD, MCPServer, drain_stderr  # noqa: E402


async def main():
    tmp = tempfile.mkdtemp(prefix="mcpb_smoke_")
    env = dict(os.environ, OUTPUT_DIR=tmp, PYTHONIOENCODING="utf-8")
    proc = await asyncio.create_subprocess_exec(
        *CMD, cwd=BUNDLE_SRC, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    asyncio.get_running_loop().create_task(drain_stderr(proc.stderr, []))
    srv = MCPServer(proc, proc.stdin)
    await srv.call("initialize", {"protocolVersion": "2026-07-28", "capabilities": {},
                                  "clientInfo": {"name": "smoke", "version": "1"}}, timeout=240)
    await srv.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    r = await srv.call("tools/call", {"name": "list_sources", "arguments": {"category": "all"}})
    text = r["result"]["content"][0]["text"]
    n = sum(1 for ln in text.splitlines() if ln.startswith(("TEXT |", "VISION |")))
    print(f"jumlah sumber: {n}")
    assert "IndoRad-VQA-test" not in text, "IndoRad-VQA-test masih ada!"
    assert "IndoRad-VQA-train" in text, "IndoRad-VQA-train hilang!"
    assert n == 15, f"harusnya 15, dapat {n}"
    print("✅ katalog 15, IndoRad-test hilang, train ada")

    r = await srv.call("tools/call", {"name": "sample_row", "arguments": {
        "category": "vision", "source_key": "IndoRad-VQA-train", "seed": 7}}, timeout=240)
    info = json.loads(r["result"]["content"][0]["text"])
    print("sample IndoRad-VQA-train ->", info["source"], "| image_available:", info["image_available"])
    inline = r["result"]["content"][1] if len(r["result"]["content"]) > 1 else None
    print("inline image:", inline.get("type") if inline else None,
          "| b64 len:", len(inline.get("data", "")) if inline else 0)
    assert inline and inline.get("data"), "IndoRad train harus ada gambar inline"
    proc.terminate()
    await asyncio.wait_for(proc.wait(), timeout=10)
    print("✅ SMOKE LULUS")


if __name__ == "__main__":
    asyncio.run(main())
