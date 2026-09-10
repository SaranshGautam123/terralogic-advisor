"""
verify_dataset.py
=================
TerraLogic Advisor — Stage 4 Dataset Split & Leakage Verification Tool

Verifies:
  1. Existence and integrity of 'data_soil/train/', 'data_soil/val/', and 'data_soil/test/'.
  2. Presence and distribution of all 4 expected classes: Alluvial, Black, Clay, Red.
  3. Image file validity (zero-byte, corrupted headers, unreadable images).
  4. Channel mode verification (ensures all images are 3-channel RGB).
  5. Intra-split duplicate check (ensures zero duplicate files within any split).
  6. Cross-split leakage check (ensures Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).
  7. Cross-class collision check (ensures no identical image appears under different classes).
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

DATA_DIR = Path("data_soil")
SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = ["Alluvial", "Black", "Clay", "Red"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_hashes(filepath: Path):
    """Computes both file-byte SHA-256 and decoded RGB pixel SHA-256."""
    h_file = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h_file.update(chunk)
    file_hash = h_file.hexdigest()

    with Image.open(filepath) as img:
        img.verify()
    with Image.open(filepath) as img:
        img.load()
        rgb = img.convert("RGB")
        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        w, h = rgb.size
        mode = img.mode

    return file_hash, pixel_hash, w, h, mode


def audit_splits():
    print("=" * 75)
    print("TERRALOGIC ADVISOR — DATASET SPLIT & LEAKAGE VERIFICATION")
    print("=" * 75)

    if not DATA_DIR.exists():
        print(f"\n[FAIL] Missing clean dataset directory: '{DATA_DIR}'")
        return False

    split_class_counts = {s: Counter() for s in SPLITS}
    split_pixel_hashes = {s: set() for s in SPLITS}
    split_file_hashes = {s: set() for s in SPLITS}
    all_pixel_hashes = defaultdict(list)

    total_images = 0
    corrupt_files = []
    non_rgb_files = []

    # 1. Scan and inspect all files
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"[FAIL] Missing split folder: '{split_dir}'")
            return False

        for cls in EXPECTED_CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"[FAIL] Missing class folder: '{cls_dir}'")
                return False

            for fpath in sorted(cls_dir.iterdir()):
                if not fpath.is_file() or fpath.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                if fpath.stat().st_size == 0:
                    corrupt_files.append((str(fpath), "Zero-byte file"))
                    continue

                try:
                    fhash, phash, w, h, mode = compute_hashes(fpath)
                    total_images += 1
                    split_class_counts[split][cls] += 1

                    if mode != "RGB":
                        non_rgb_files.append((str(fpath), mode))

                    split_file_hashes[split].add(fhash)
                    split_pixel_hashes[split].add(phash)
                    all_pixel_hashes[phash].append((split, cls, fpath.name))
                except Exception as e:
                    corrupt_files.append((str(fpath), str(e)))

    # 2. Display distribution table
    print("\n--- DATASET SPLIT AUDIT SUMMARY ---")
    print(f"{'Class Name':<14} {'Train (70%)':<14} {'Val (15%)':<12} {'Test (15%)':<12} {'Total Unique':<12}")
    print("-" * 64)
    for cls in EXPECTED_CLASSES:
        tr = split_class_counts["train"][cls]
        va = split_class_counts["val"][cls]
        te = split_class_counts["test"][cls]
        tot = tr + va + te
        print(f"{cls:<14} {tr:<14} {va:<12} {te:<12} {tot:<12}")
    print("-" * 64)
    tot_tr = sum(split_class_counts["train"].values())
    tot_va = sum(split_class_counts["val"].values())
    tot_te = sum(split_class_counts["test"].values())
    print(f"{'TOTAL':<14} {tot_tr:<14} {tot_va:<12} {tot_te:<12} {total_images:<12}")

    # 3. Check for integrity failures
    has_error = False

    print("\n--- INTEGRITY CHECKS ---")
    print(f"[*] Total files verified          : {total_images}")
    print(f"[*] Corrupted / unreadable files   : {len(corrupt_files)}")
    print(f"[*] Non-RGB mode images            : {len(non_rgb_files)}")

    if corrupt_files:
        print("[FAIL] Corrupt files detected!")
        for cp, reason in corrupt_files[:5]:
            print(f"    - {cp}: {reason}")
        has_error = True

    if non_rgb_files:
        print("[FAIL] Non-RGB files detected!")
        has_error = True

    # 4. Check for cross-split leakage
    leak_tr_va = split_pixel_hashes["train"].intersection(split_pixel_hashes["val"])
    leak_tr_te = split_pixel_hashes["train"].intersection(split_pixel_hashes["test"])
    leak_va_te = split_pixel_hashes["val"].intersection(split_pixel_hashes["test"])

    print("\n--- LEAKAGE VERIFICATION ---")
    print(f"[*] Train intersect Val overlap  : {len(leak_tr_va)} (Expected: 0)")
    print(f"[*] Train intersect Test overlap : {len(leak_tr_te)} (Expected: 0)")
    print(f"[*] Val intersect Test overlap   : {len(leak_va_te)} (Expected: 0)")

    if leak_tr_va or leak_tr_te or leak_va_te:
        print("[FAIL] Exact pixel leakage detected between splits!")
        has_error = True

    # 5. Check for cross-class collisions
    cross_class = [ph for ph, recs in all_pixel_hashes.items() if len(set(r[1] for r in recs)) > 1]
    print(f"[*] Cross-class duplicate collisions: {len(cross_class)} (Expected: 0)")
    if cross_class:
        print("[FAIL] Cross-class duplicate collisions detected!")
        has_error = True

    # 6. Check for intra-split duplicate images
    duplicate_instances = [ph for ph, recs in all_pixel_hashes.items() if len(recs) > 1]
    print(f"[*] Duplicate pixel instances: {len(duplicate_instances)} (Expected: 0)")
    if duplicate_instances:
        print("[FAIL] Duplicate images found inside dataset splits!")
        has_error = True

    print("\n" + "=" * 75)
    if not has_error and total_images > 0:
        print("[PASS] ALL DATASET VERIFICATION & LEAKAGE CHECKS PASSED!")
        print("=" * 75)
        return True
    else:
        print("[FAIL] DATASET VERIFICATION FAILED. Review errors above.")
        print("=" * 75)
        return False


if __name__ == "__main__":
    passed = audit_splits()
    sys.exit(0 if passed else 1)
