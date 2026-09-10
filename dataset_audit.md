# Soil Image Dataset Audit & Cleaning Report
**Dataset:** 4-Class Soil Types Dataset (Kaggle: `jhislainematchouath/soil-types-dataset`)  
**Stage:** Stage 4.1 — Dataset Cleaning, Deduplication, Quarantine & Leakage-Safe Splitting  
**Execution Seed:** `42` (Deterministic)  
**Raw Source Directory:** `data_soil_raw/` (Preserved completely untouched)  
**Clean Target Directory:** `data_soil/`

---

## 1. Raw Dataset Summary & Defect Breakdown

| Metric | Measured Count | Notes |
|---|---|---|
| **Total Raw Images** | **1555** | Initial files in `data_soil_raw/` |
| **Valid Images** | **1555** | Readable and valid image headers via PIL |
| **Corrupt / Unreadable Images** | **0** | Zero corrupt files found |
| **Unique Raw File Hashes (SHA-256)** | **699** | File container bytes |
| **Unique Decoded Pixel Hashes (SHA-256)** | **697** | Decoded uncompressed RGB byte streams |
| **Same-Label Duplicate Copies Quarantined** | **852** | Exact copies within same class label |
| **Cross-Class Conflicting Files Quarantined** | **8** | Identical pixel data across different classes (2 distinct hashes) |
| **Final Clean Unique Images Retained** | **695** | Pristine, deduplicated, single-label image pool |

---

## 2. Raw vs. Clean Class Distribution

| Class Name | Raw Image Count | Quarantined Duplicates & Conflicts | Final Clean Image Count | Retention % |
|---|---|---|---|---|
| **Alluvial** | 576 | 285 | **291** | 50.5% |
| **Black** | 344 | 223 | **121** | 35.2% |
| **Clay** | 262 | 144 | **118** | 45.0% |
| **Red** | 373 | 208 | **165** | 44.2% |
| **TOTAL** | **1555** | **860** | **695** | **44.7%** |

---

## 3. Cross-Class Duplicate Conflict Log

Two distinct pixel streams occurred under multiple conflicting class labels. All instances have been purged from the clean training pool:

1. **Pixel Hash `2ee340c07494`**: Conflicted between **Alluvial** and **Black** (4 files total):
   - `Alluvial/alluvial_0251.jpg`
   - `Black/black_0101.jpg`
   - `Black/black_0218.jpg`
   - `Black/black_0327.jpg`
2. **Pixel Hash `44a990b5ee1e`**: Conflicted between **Clay** and **Red** (4 files total):
   - `Clay/clay_0001.png`
   - `Clay/clay_0026.png`
   - `Clay/clay_0068.png`
   - `Red/red_0118.png`

---

## 4. Final Stratified Split Distribution (70% / 15% / 15%)

Partitioned using stratified random sampling with fixed seed `42`:

| Class Name | Train Split (70%) | Validation Split (15%) | Test Split (15%) | Total Unique Images |
|---|---|---|---|---|
| **Alluvial** | 203 (69.8%) | 44 (15.1%) | 44 (15.1%) | **291** |
| **Black** | 84 (69.4%) | 18 (14.9%) | 19 (15.7%) | **121** |
| **Clay** | 82 (69.5%) | 18 (15.3%) | 18 (15.3%) | **118** |
| **Red** | 115 (69.7%) | 25 (15.2%) | 25 (15.2%) | **165** |
| **TOTAL** | **484** (69.6%) | **105** (15.1%) | **106** (15.3%) | **695** (100.0%) |

---

## 5. Leakage Verification Results

Exact cryptographic verification of decoded RGB pixel arrays between all partition pairs:

- **Train intersect Validation Overlap:** `0` hashes (`PASS`)
- **Train intersect Test Overlap:** `0` hashes (`PASS`)
- **Validation intersect Test Overlap:** `0` hashes (`PASS`)
- **Cross-Class Duplicate Overlap:** `0` hashes (`PASS`)

*Conclusion:* **Zero exact data leakage across splits.**

---

## 6. Limitations & Viva Notes

1. **Perceptual Near-Duplicates**: Exact pixel and file hashing guarantees that no duplicate images exist across splits. However, web-scraped datasets may still contain near-duplicate frames (e.g. photos taken seconds apart from slightly shifted angles).
2. **Scientific Scope**: Visual classification identifies physical soil types (morphological features, surface color, coarse granularity). It does **not** determine exact chemical Nitrogen, Phosphorus, Potassium concentrations, or exact pH, which are handled independently by the Numeric Soil Health Advisor.
