#!/usr/bin/env python3
"""Prüft, ob Original und abliterierte Variante eines Paares gleich quantisiert sind (GGUF-Metadaten).

Verglichen werden general.file_type, general.quantization_version und ob Kalibrierdaten (quantize.imatrix.*)
genutzt wurden. Nur Standardbibliothek; liest nur den Kopf der Dateien.
  scripts/check_gguf_pairs.py $MODELS_DIR [--out runs/_system/gguf_pairs.json]
Exit 1, wenn ein Paar nicht zusammenpasst.
"""
import json
import struct
import sys
from pathlib import Path

SIZES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
FILE_TYPES = {15: "Q4_K_M", 14: "Q4_K_S", 7: "Q8_0", 1: "F16", 32: "BF16"}


def meta(path: Path) -> dict:
    with path.open("rb") as fh:
        def rd(fmt):
            return struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))[0]

        def rs():
            return fh.read(rd("Q")).decode("utf-8", "replace")

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = rd("I"), rd("Q")
                if et == 8:
                    for _ in range(n):
                        rs()
                else:
                    fh.seek(n * struct.calcsize("<" + SIZES[et]), 1)
                return None
            return rd(SIZES[t])

        assert fh.read(4) == b"GGUF", f"{path}: kein GGUF"
        rd("I"), rd("Q")
        out = {}
        for _ in range(rd("Q")):
            k = rs()
            out[k] = val(rd("I"))
        return out


def main() -> int:
    root = Path(sys.argv[1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    info = {}
    for gguf in sorted(root.glob("*/*.gguf")):
        if "mmproj" in gguf.name:
            continue
        m = meta(gguf)
        info[gguf.parent.name] = {
            "file": gguf.name,
            "file_type": FILE_TYPES.get(m.get("general.file_type"), m.get("general.file_type")),
            "quantization_version": m.get("general.quantization_version"),
            "imatrix": any(k.startswith("quantize.imatrix") for k in m),
            "architecture": m.get("general.architecture"),
            "source": m.get("general.source.url"),
        }
    ok = True
    pairs = []
    for alias in sorted(a for a in info if not a.endswith("-abliterated")):
        abl = alias + "-abliterated"
        if abl not in info:
            continue
        a, b = info[alias], info[abl]
        diff = [k for k in ("file_type", "quantization_version", "imatrix", "architecture") if a[k] != b[k]]
        pairs.append({"original": alias, "abliterated": abl, "match": not diff, "differences": diff})
        print(f"{'OK    ' if not diff else 'FEHLER'} {alias} <-> {abl}: {a['file_type']}, qv {a['quantization_version']}, "
              f"imatrix {a['imatrix']}" + (f" | Unterschied: {diff}" if diff else ""))
        ok &= not diff
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"models": info, "pairs": pairs}, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
