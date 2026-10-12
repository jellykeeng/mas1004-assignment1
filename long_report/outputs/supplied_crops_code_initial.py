"""Compare unchanged student-provided JPEG/PNG crops with their original photos.

Run through uv from the repository folder. This copies the provided file bytes
and applies only the existing model preprocessing; it never chooses a crop.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import onnxruntime as ort
from PIL import Image

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
from crop_experiment import (CROP, MEAN, RESIZE, STD, path_from_root,
                             predict, recorded_path, save_rgb_png, sha256)


def image_record(path):
    """Describe decoded pixels and embedded metadata without changing either."""
    with Image.open(path) as image:
        image.load()
        info = {"path": recorded_path(path), "sha256": sha256(path),
                "bytes": path.stat().st_size, "format": image.format,
                "mode": image.mode, "size": list(image.size),
                "decoded_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "rgb_pixel_sha256": hashlib.sha256(
                    image.convert("RGB").tobytes()).hexdigest(),
                "metadata_keys": sorted(image.info)}
        for key in ("icc_profile", "exif"):
            value = image.info.get(key)
            info[f"has_{key}"] = bool(value)
            if isinstance(value, bytes):
                info[f"{key}_sha256"] = hashlib.sha256(value).hexdigest()
        return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/supplied_crops.json")
    parser.add_argument("--output", default="results/supplied_crops.json")
    args = parser.parse_args()
    config_path = path_from_root(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = path_from_root(config.get("model", "docs/model.onnx"))
    metadata_path = path_from_root(config.get("metadata", "docs/model.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    expected = {"resize": RESIZE, "crop": CROP,
                "mean": list(MEAN), "std": list(STD)}
    if any(metadata["input"].get(key) != value for key, value in expected.items()):
        raise ValueError("Export metadata does not match src/data.py preprocessing.")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Model labels must be nonempty and unique.")

    output_dir = path_from_root(config.get("output_dir", "data/changed/supplied_crops"))
    output_path = path_from_root(args.output)
    if not output_dir.is_relative_to(ROOT / "data" / "changed" / "supplied_crops"):
        raise ValueError("output_dir must be under data/changed/supplied_crops.")
    if not output_path.is_relative_to(ROOT / "results"):
        raise ValueError("--output must be under results.")
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list of supplied files.")
    ids = [case["id"] for case in cases]
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
           for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Case ids must be unique names using letters, numbers, _ or -.")

    # Read and validate every supplied choice before writing any outputs.
    sources = [path_from_root(case["source"]) for case in cases]
    provided = [path_from_root(case["provided"]) for case in cases]
    protected = {config_path, model_path, metadata_path, *sources, *provided,
                 ROOT / "src" / "crop_experiment.py", ROOT / "src" / "data.py"}
    source_info, provided_info = [], []
    for case, source, supplied in zip(cases, sources, provided):
        if case["true_label"] not in labels:
            raise ValueError(f"Unknown true_label for {case['id']}: {case['true_label']}")
        source_info.append(image_record(source))
        supplied_info = image_record(supplied)
        if supplied_info["format"] not in {"PNG", "JPEG"}:
            raise ValueError(f"{case['id']}: provided must be an actual JPEG or PNG file.")
        provided_info.append(supplied_info)
        case_dir = (output_dir / case["id"]).resolve()
        if not case_dir.is_relative_to(output_dir):
            raise ValueError("A case directory would leave the allowed output folder.")
        for name in ("original.png", "provided.png", "provided.jpg", "original_input.png", "changed_input.png"):
            if (case_dir / name).resolve() in protected:
                raise ValueError("An output would overwrite a source/config/model/code file.")
    if output_path in protected:
        raise ValueError("JSON output would overwrite a source/config/model/code file.")

    initial_hashes = {path: sha256(path) for path in protected}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print("Student-provided crop comparison: existing ONNX model, CPU, no training")
    print("Provided files are copied byte for byte; no extra crop, resize or image generation")
    print("Only model inputs use the existing RGB/bilinear resize/centre-crop preprocessing")
    print(f"model: {recorded_path(model_path)}; SHA256 {initial_hashes[model_path]}")
    print("labels: " + ", ".join(labels))
    print(f"Pending supplied files (not measured): {len(config.get('pending_cases', []))}")
    for case, source, supplied, original_info, supplied_info in zip(
            cases, sources, provided, source_info, provided_info):
        case_dir = output_dir / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        copied_path = case_dir / ("provided.png" if supplied_info["format"] == "PNG" else "provided.jpg")
        shutil.copyfile(supplied, copied_path)
        assert supplied.read_bytes() == copied_path.read_bytes(), "Provided file bytes changed."
        copied_info = image_record(copied_path)
        assert copied_info["sha256"] == supplied_info["sha256"]
        assert all(copied_info[key] == supplied_info[key] for key in (
            "format", "mode", "size", "decoded_pixel_sha256", "rgb_pixel_sha256", "metadata_keys"))

        with Image.open(source) as image:
            original = image.convert("RGB")
        original_copy = save_rgb_png(original, case_dir / "original.png")
        before, before_square = predict(session, metadata, original, case["true_label"])
        with Image.open(copied_path) as image:
            after, after_square = predict(session, metadata, image, case["true_label"])
        before["input_image"] = save_rgb_png(before_square, case_dir / "original_input.png")
        after["input_image"] = save_rgb_png(after_square, case_dir / "changed_input.png")
        records.append({"id": case["id"], "true_label": case["true_label"],
                        "source": recorded_path(source), "source_details": original_info,
                        "source_sha256": initial_hashes[source], "provided_source": supplied_info,
                        "original_image": original_copy, "changed_image": copied_info,
                        "provided_image": copied_info,
                        "provided_bytes_unchanged": True, "provided_pixels_unchanged": True,
                        "provided_embedded_metadata_unchanged": True,
                        "input_matches_prepare_image": True, "before": before, "after": after})
        print(f"\n{case['id']}: true {case['true_label']}")
        print(f"source: {recorded_path(source)}; SHA256 {initial_hashes[source]}")
        print(f"provided: {recorded_path(supplied)}; SHA256 {initial_hashes[supplied]}")
        print(f"original size {tuple(original_info['size'])}; provided size {tuple(supplied_info['size'])}")
        for stage, result in (("before", before), ("after", after)):
            print(f"{stage}: {result['pred_label']} {result['pred_probability']:.1%}; "
                  f"true-label {result['true_probability']:.1%}; correct {result['correct']}")
            print("  " + "; ".join(f"{label} {result['probabilities'][label]:.1%}" for label in labels))
        print("Provided file bytes/pixels/embedded metadata unchanged: True")
        print("224-pixel inputs match prepare_image: True")

    preserved = all(sha256(path) == digest for path, digest in initial_hashes.items())
    assert preserved, "A source, provided file, config, metadata, model or helper changed."
    accuracy = {}
    for stage in ("before", "after"):
        correct = sum(record[stage]["correct"] for record in records)
        accuracy[stage] = {"correct": correct, "total": len(records),
                           "percent": 100 * correct / len(records)}
        print(f"\n{stage} selected-case top-1 accuracy: {correct}/{len(records)} = "
              f"{accuracy[stage]['percent']:.1f}%")
    print("Source/provided/config/metadata/model/helper SHA256 unchanged: True")
    result = {"config": recorded_path(config_path), "config_sha256": initial_hashes[config_path],
              "model": recorded_path(model_path), "model_sha256": initial_hashes[model_path],
              "metadata": recorded_path(metadata_path), "metadata_sha256": initial_hashes[metadata_path],
              "labels": labels, "preprocessing": metadata["input"],
              "provider": "CPUExecutionProvider", "inferences_per_case": {"before": 1, "after": 1},
              "provided_processing": "byte copy only; no crop/resampling/generation",
              "student_request_verbatim": config.get("student_request_verbatim"),
              "selection": config.get("selection"), "pending_cases": config.get("pending_cases", []),
              "protected_files_unchanged": preserved,
              "protected_sha256": {recorded_path(path): digest for path, digest in initial_hashes.items()},
              "accuracy": accuracy, "cases": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {recorded_path(output_path)}")


if __name__ == "__main__":
    main()
