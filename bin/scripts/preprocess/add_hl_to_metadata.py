#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def add_hl_field(
    input_jsonl: Path,
    output_jsonl: Path,
    hl_root: Path,
    field_name: str = "hl_npz",
    strict: bool = False,
    skip_missing: bool = False,
):
    """
    Read jsonl, add one field pointing to HL npz path:
      hl_path = hl_root / (video_id + ".npz")

    strict=True:  any missing -> raise
    skip_missing=True: drop those lines with missing HL
    """
    hl_root = Path(hl_root)
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = 0
    kept = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, \
         output_jsonl.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Processing {input_jsonl.name}"):
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)

            if "video_id" not in item:
                raise KeyError(f"{input_jsonl}: line {total} missing required key 'video_id'")

            # video_id example: "task/0"
            rel = Path(item["video_id"]).with_suffix(".npz")
            hl_path = hl_root / rel

            if not hl_path.exists():
                missing += 1
                msg = f"Missing HL: video_id={item['video_id']} -> {hl_path}"
                if strict:
                    raise FileNotFoundError(msg)
                if skip_missing:
                    continue
                # keep but mark as None
                item[field_name] = None
                item[field_name + "_missing"] = True
            else:
                item[field_name] = str(hl_path)
                item[field_name + "_missing"] = False
                kept += 1

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "input": str(input_jsonl),
        "output": str(output_jsonl),
        "total": total,
        "kept_with_hl": kept,
        "missing_hl": missing,
        "hl_root": str(hl_root),
    }


def main():
    ap = argparse.ArgumentParser("Add HL npz path to existing metadata jsonl")
    ap.add_argument("--metadata_dir", required=True, help="Directory containing train_metadata.jsonl/test_metadata.jsonl")
    ap.add_argument("--hl_root", required=True, help="Root dir of HL npz files, e.g. EgoDex_HL/")
    ap.add_argument("--output_dir", required=True, help="Where to write new jsonl files")
    ap.add_argument("--field_name", default="hl_npz", help="Field name to store HL path")
    ap.add_argument("--strict", action="store_true", help="Fail if any HL file missing")
    ap.add_argument("--skip_missing", action="store_true", help="Drop items with missing HL instead of keeping them")
    args = ap.parse_args()

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # support either train/test, or any *.jsonl in dir
    input_files = []
    for name in ["train_metadata.jsonl", "test_metadata.jsonl"]:
        p = metadata_dir / name
        if p.exists():
            input_files.append(p)

    if not input_files:
        input_files = sorted(metadata_dir.glob("*.jsonl"))
        if not input_files:
            raise FileNotFoundError(f"No jsonl found in {metadata_dir}")

    summaries = []
    for inp in input_files:
        outp = output_dir / inp.name
        summaries.append(
            add_hl_field(
                inp, outp,
                hl_root=Path(args.hl_root),
                field_name=args.field_name,
                strict=args.strict,
                skip_missing=args.skip_missing,
            )
        )

    # print summaries
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()