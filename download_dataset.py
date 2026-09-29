"""
download_dataset.py
===================
TerraLogic Advisor — Automated Soil Image Dataset Downloader

Downloads and prepares the 4-class Soil Types dataset from Kaggle:
  - Alluvial Soil
  - Black Soil
  - Clay Soil
  - Red Soil

Organizes images into clean directory structure:
  data_soil/
    ├── Alluvial/
    ├── Black/
    ├── Clay/
    └── Red/
"""

import os
import sys
import shutil
import zipfile

DATASET_SLUG = "prasanshasatpathy/soil-types"
FALLBACK_SLUG = "jhislainematchouath/soil-types-dataset"
TARGET_DIR = "data_soil"
TEMP_DOWNLOAD_DIR = "temp_dataset_download"

EXPECTED_CLASSES = ["Alluvial", "Black", "Clay", "Red"]


def ensure_kaggle():
    """Ensure kaggle library is installed and credentials exist."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[+] Kaggle API initialized and authenticated successfully.")
        return api
    except Exception as e:
        print(f"[!] Kaggle API Authentication Error: {e}")
        print("\nPlease ensure '~/.kaggle/kaggle.json' exists with valid credentials.")
        return None


def download_and_extract(api):
    """Downloads dataset from Kaggle and organizes into data_soil/."""
    print("=" * 65)
    print("TERRALOGIC ADVISOR — AUTOMATIC DATASET DOWNLOAD")
    print("=" * 65)

    if os.path.exists(TEMP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

    downloaded = False
    for slug in [DATASET_SLUG, FALLBACK_SLUG]:
        try:
            print(f"\n[*] Attempting download from Kaggle: {slug} ...")
            api.dataset_download_files(slug, path=TEMP_DOWNLOAD_DIR, unzip=True)
            print(f"[+] Download and extraction of {slug} succeeded!")
            downloaded = True
            break
        except Exception as e:
            print(f"[!] Failed downloading {slug}: {e}")

    if not downloaded:
        print("[ERROR] Could not download soil dataset from Kaggle.")
        return False

    # Create destination directories
    for cls in EXPECTED_CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

    # Walk through extracted files and match to expected classes
    print("\n[*] Organizing images into clean 4-class structure in data_soil/...")
    copied_counts = {cls: 0 for cls in EXPECTED_CLASSES}

    for root, dirs, files in os.walk(TEMP_DOWNLOAD_DIR):
        folder_name = os.path.basename(root).lower()

        target_class = None
        for cls in EXPECTED_CLASSES:
            if cls.lower() in folder_name:
                target_class = cls
                break

        if target_class is not None:
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    src_path = os.path.join(root, f)
                    dest_fname = f"{target_class.lower()}_{copied_counts[target_class] + 1}{ext}"
                    dest_path = os.path.join(TARGET_DIR, target_class, dest_fname)
                    
                    # Copy if not already existing
                    if not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                    copied_counts[target_class] += 1

    # Cleanup temp folder
    try:
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
        print("[+] Cleaned up temporary download cache.")
    except Exception:
        pass

    # Print summary
    print("\n" + "-" * 65)
    print("DATASET PREPARATION SUMMARY:")
    print("-" * 65)
    total_imgs = 0
    for cls in EXPECTED_CLASSES:
        print(f"  - {cls:<12}: {copied_counts[cls]} images")
        total_imgs += copied_counts[cls]
    print("-" * 65)
    print(f"  TOTAL: {total_imgs} images organized into '{TARGET_DIR}/'")
    print("=" * 65)

    return total_imgs > 0


if __name__ == "__main__":
    api = ensure_kaggle()
    if api is None:
        sys.exit(1)

    success = download_and_extract(api)
    if not success:
        sys.exit(1)

