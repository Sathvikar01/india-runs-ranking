"""Fetch the Phi-3.5-mini int4 GGUF model for the local LLM fallback.

This is a *build-time-only* step. It requires network access and produces
`artifacts/phi3.5-mini-int4.gguf` (≈ 2.2 GB). The artifact is committed
neither to git nor to the 5 GB submission bundle (we ship only the
post-build outputs and re-fetch at Stage 3 if needed).

Run with:
    python scripts/fetch_local_llm.py --out artifacts/

Wall-clock on a 16 GB CPU-only laptop: 5-10 minutes (1-2 MB/s typical).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("fetch_local_llm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Phi-3.5-mini-instruct Q4_K_M (≈ 2.2 GB). The "Q4_K_M" quantization is the
# standard int4 quality/size sweet spot for 3.8B-param models. We use the
# bartowski republish (no auth required for direct downloads).
DEFAULT_URL = (
    "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/"
    "resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
)
DEFAULT_FILENAME = "phi3.5-mini-int4.gguf"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch the Phi-3.5-mini int4 GGUF model.")
    p.add_argument("--out", default="artifacts", help="Output directory (default: artifacts)")
    p.add_argument("--url", default=DEFAULT_URL, help="GGUF download URL")
    p.add_argument("--filename", default=DEFAULT_FILENAME, help="Local filename")
    p.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename

    if out_path.exists() and not args.force:
        log.info(
            "Model already present at %s. Use --force to re-download.",
            out_path,
        )
        return 0

    log.info("Downloading %s -> %s", args.url, out_path)
    import urllib.request

    tmp_path = out_path.with_suffix(".gguf.part")
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        log.info("Total bytes: %s", f"{total/1e9:.2f} GB" if total else "unknown")
        written = 0
        last_log = 0
        with tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total and written - last_log > 50_000_000:
                    log.info("  %.1f%% (%d / %d)", 100 * written / total, written, total)
                    last_log = written
    tmp_path.replace(out_path)
    log.info("Downloaded %s (sha256=%s)", out_path, _sha256(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
