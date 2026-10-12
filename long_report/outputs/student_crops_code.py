"""Measure two student-selected pixel crops with the existing ONNX model.

Run from the repository folder through uv. This only writes lossless copies
under data/changed and a results JSON; it never trains or changes a source.
Crop boxes use Pillow's decoded pixels: left/top included, right/bottom excluded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
from data import CROP, MEAN, RESIZE, STD, prepare_image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_from_root(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def recorded_path(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def save_rgb_png(image, path):
    # Creating a fresh image omits ICC/EXIF metadata without changing RGB pixels.
    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")
    with Image.open(path) as saved:
        assert saved.mode == "RGB" and "icc_profile" not in saved.info
        assert np.array_equal(np.asarray(saved), pixels), f"PNG pixels changed: {path}"
    return {"path": recorded_path(path), "sha256": sha256(path),
            "size": list(image.size), "rgb_pixels_unchanged": True,
            "icc_profile_omitted": True}


def model_input(image):
    # Save the actual RGB resize/centre-crop, not a normalization reversal.
    square = F.center_crop(
        F.resize(image.convert("RGB"), RESIZE,
                 interpolation=InterpolationMode.BILINEAR), [CROP, CROP])
    prepared = prepare_image(image)
    from_square = F.normalize(F.to_tensor(square), MEAN, STD).numpy()
    assert prepared.dtype == np.float32 and prepared.shape == (3, CROP, CROP)
    assert np.array_equal(prepared, from_square), "224-pixel view differs from prepare_image"
    return prepared, square


def predict(session, metadata, image, true_label):
    prepared, square = model_input(image)
    # Exactly one inference for this image.
    logits = np.asarray(session.run(
        [metadata["output"]["name"]],
        {metadata["input"]["name"]: prepared[None]})[0][0], dtype=np.float32)
    labels = metadata["labels"]
    if logits.shape != (len(labels),) or not np.isfinite(logits).all():
        raise ValueError("The ONNX model did not return finite logits for every label.")
    weights = np.exp(logits.astype(np.float64) - float(logits.max()))
    probabilities = weights / weights.sum()
    predicted = int(np.argmax(logits))
    true_index = labels.index(true_label)
    return {"logits": logits.tolist(),
            "probabilities": dict(zip(labels, probabilities.tolist())),
            "pred_index": predicted, "pred_label": labels[predicted],
            "pred_probability": float(probabilities[predicted]),
            "true_probability": float(probabilities[true_index]),
            "correct": labels[predicted] == true_label}, square


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/student_crops.json")
    parser.add_argument("--output", default="results/student_crops.json")
    args = parser.parse_args()
    config_path = path_from_root(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = path_from_root(config["model"])
    metadata_path = path_from_root(config["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    expected = {"resize": RESIZE, "crop": CROP,
                "mean": list(MEAN), "std": list(STD)}
    if any(metadata["input"].get(key) != value for key, value in expected.items()):
        raise ValueError("Export metadata does not match src/data.py preprocessing.")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Model labels must be nonempty and unique.")

    output_dir = path_from_root(config["output_dir"])
    output_path = path_from_root(args.output)
    if not output_dir.is_relative_to(ROOT / "data" / "changed"):
        raise ValueError("output_dir must be under data/changed, outside training/evaluation.")
    if not output_path.is_relative_to(ROOT / "results"):
        raise ValueError("--output must be under results.")
    cases = config["cases"]
    if len(cases) != 2:
        raise ValueError("This student-selected experiment requires exactly two cases.")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or any(
            not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
            for value in ids):
        raise ValueError("Case ids must be unique names using letters, numbers, _ or -.")

    # Validate all choices before writing any copies; there is no coordinate search.
    source_paths = [path_from_root(case["source"]) for case in cases]
    protected_paths = {config_path, model_path, metadata_path, *source_paths}
    for case, source in zip(cases, source_paths):
        if case["true_label"] not in labels:
            raise ValueError(f"Unknown true_label for {case['id']}: {case['true_label']}")
        box = case["box"]
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box)):
            raise ValueError(f"{case['id']}: box must contain four integer pixel coordinates.")
        with Image.open(source) as image:
            width, height = image.size
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{case['id']}: box {box} is outside image size {(width, height)}.")
        for name in ("original.png", "changed.png", "before_input.png", "after_input.png"):
            if (output_dir / case["id"] / name).resolve() in protected_paths:
                raise ValueError("A copy would overwrite a protected source/config/model file.")
    if output_path in protected_paths:
        raise ValueError("JSON output would overwrite a protected source/config/model file.")

    initial_hashes = {path: sha256(path) for path in protected_paths}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print("Student-selected crop experiment: existing ONNX model, CPU, no training")
    print(f"model: {recorded_path(model_path)}; SHA256 {initial_hashes[model_path]}")
    print("Coordinates: decoded original pixels, left/top included, right/bottom excluded")
    print("labels: " + ", ".join(labels))
    for case, source in zip(cases, source_paths):
        case_dir = output_dir / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            original = image.convert("RGB")
        cropped = original.crop(tuple(case["box"]))
        original_copy = save_rgb_png(original, case_dir / "original.png")
        changed_copy = save_rgb_png(cropped, case_dir / "changed.png")
        before, before_square = predict(session, metadata, original, case["true_label"])
        with Image.open(case_dir / "changed.png") as saved_crop:
            assert np.array_equal(np.asarray(saved_crop), np.asarray(cropped))
            after, after_square = predict(session, metadata, saved_crop, case["true_label"])
        before["input_image"] = save_rgb_png(before_square, case_dir / "before_input.png")
        after["input_image"] = save_rgb_png(after_square, case_dir / "after_input.png")
        record = {"id": case["id"], "source": recorded_path(source),
                  "source_sha256": initial_hashes[source], "true_label": case["true_label"],
                  "box": case["box"], "original_size": list(original.size),
                  "cropped_size": list(cropped.size), "original_image": original_copy,
                  "changed_image": changed_copy, "crop_exact_pixels": True,
                  "input_matches_prepare_image": True, "before": before, "after": after}
        records.append(record)
        print(f"\n{case['id']}: {recorded_path(source)}; true {case['true_label']}")
        print(f"source SHA256 {initial_hashes[source]}")
        print(f"box {case['box']}; size {original.size} -> {cropped.size}")
        for stage, result in (("before", before), ("after", after)):
            print(f"{stage}: {result['pred_label']} {result['pred_probability']:.1%}; "
                  f"true-label {result['true_probability']:.1%}; correct {result['correct']}")
            print("  " + "; ".join(f"{label} {result['probabilities'][label]:.1%}"
                                    for label in labels))
        print("Crop RGB pixels exact: True; 224-pixel inputs match prepare_image: True")

    preserved = all(sha256(path) == digest for path, digest in initial_hashes.items())
    assert preserved, "A source, config, metadata or model changed during measurement."
    accuracy = {}
    for stage in ("before", "after"):
        correct = sum(record[stage]["correct"] for record in records)
        accuracy[stage] = {"correct": correct, "total": len(records),
                           "percent": 100 * correct / len(records)}
        print(f"\n{stage} top-1 accuracy: {correct}/{len(records)} = "
              f"{accuracy[stage]['percent']:.1f}%")
    print("Source/config/metadata/model SHA256 unchanged: True")
    result = {"config": recorded_path(config_path), "config_sha256": initial_hashes[config_path],
              "model": recorded_path(model_path), "model_sha256": initial_hashes[model_path],
              "metadata": recorded_path(metadata_path),
              "metadata_sha256": initial_hashes[metadata_path], "labels": labels,
              "preprocessing": metadata["input"], "provider": "CPUExecutionProvider",
              "coordinate_convention": "decoded pixels; left/top inclusive, right/bottom exclusive",
              "inferences_per_case": {"before": 1, "after": 1},
              "protected_files_unchanged": preserved, "accuracy": accuracy, "cases": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {recorded_path(output_path)}")


if __name__ == "__main__":
    main()
