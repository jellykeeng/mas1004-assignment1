"""Append a reviewed supplement without overwriting downloaded originals.

Run from the repository through uv. The manifest is a JSON object with a
``decisions`` list or a mapping keyed by category/filename. Each row contains
source_path, keep, reason, and optional duplicate_of fields.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-output", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    review = json.loads(args.manifest.read_text())
    if "decisions" in review:
        decisions = {row["path"]: row for row in review["decisions"]}
    else:
        decisions = review
    raw_folders = {
        unicodedata.normalize("NFC", folder.name): folder
        for folder in (root / "data/raw").iterdir() if folder.is_dir()
    }
    planned = []
    for key, row in decisions.items():
        category, filename = key.split("/")
        raw_dir = raw_folders[unicodedata.normalize("NFC", category)]
        raw = raw_dir / filename
        clean = root / "data/clean" / raw_dir.name / filename
        source = Path(row["source_path"])
        if not source.is_absolute():
            source = args.manifest.parent / source
        if raw.exists() or clean.exists():
            raise FileExistsError(f"Refuse to overwrite: {key}")
        if not source.is_file():
            raise FileNotFoundError(source)
        planned.append((key, row, source, raw, clean))
    if not args.download_output.is_file():
        raise FileNotFoundError(args.download_output)
    if args.features is not None and not args.features.is_file():
        raise FileNotFoundError(args.features)

    raw_hashes_path = root / "results/raw_hashes_before_cleaning.json"
    old_hashes = json.loads(raw_hashes_path.read_text())
    groups = defaultdict(list)
    for key, row, source, raw, clean in planned:
        shutil.copy2(source, raw)
        shutil.copy2(raw, clean)
        if row["keep"]:
            continue
        if row.get("duplicate_of"):
            reason = "동일 원본·자세의 크기/압축/자막/UI 변형인 근접 중복"
        elif "같은 비중" in row["reason"] or "동일 비중" in row["reason"]:
            reason = "대상 캐릭터가 주된 대상이 아니거나 여러 인물이 대등하게 등장"
        else:
            reason = "다른 캐릭터이거나 대상 캐릭터를 식별할 수 없음"
        groups[reason].append(str(clean.relative_to(root / "data/clean")))
    for reason, files in groups.items():
        subprocess.run(
            ["uv", "run", "--frozen", "python", "src/clean.py", "remove",
             *files, "--reason", reason], check=True,
        )
    (root / "results/supplement_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2)
    )
    shutil.copy2(args.download_output, root / "results/collect_supplement.txt")
    if args.features is not None:
        cache_path = root / "results/cleaning/features.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            rows = dict(zip(cache["keys"].tolist(), cache["rows"]))
        with np.load(args.features, allow_pickle=False) as extra:
            rows.update(dict(zip(extra["keys"].tolist(), extra["rows"])))
        np.savez(cache_path, keys=np.array(list(rows)), rows=np.stack(list(rows.values())))
        print("Combined cached image feature rows:", len(rows))
    for key, expected in old_hashes.items():
        actual = hashlib.sha256((root / "data/raw" / key).read_bytes()).hexdigest()
        assert actual == expected, f"Original changed: {key}"
    kept = sum(row["keep"] for row in decisions.values())
    print("Supplement downloaded:", len(decisions), "kept:", kept,
          "removed:", len(decisions) - kept)
    print("Initial raw hashes unchanged:", len(old_hashes))


if __name__ == "__main__":
    main()
