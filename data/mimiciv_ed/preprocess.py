"""
Preprocessing script for MIMIC-IV-ED.

Merges selected variables from edstays, diagnosis, and triage into a single
flat CSV. The merge key is (subject_id, stay_id).

Output
------
data/mimiciv_ed/mimiciv_ed_merged.csv

Notes
-----
- diagnosis has one row per diagnosis per stay (seq_num >= 1). Only the
  primary diagnosis (seq_num == 1) is kept to keep the output one-row-per-stay.
- All other tables are already one-row-per-stay.
"""

from __future__ import annotations

import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ED_DIR = pathlib.Path(__file__).parent / "mimic-iv-ed-2.2" / "ed"
OUT_PATH = pathlib.Path(__file__).parent / "mimiciv_ed_merged.csv"

# ---------------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------------
print("Loading edstays...")
edstays = pd.read_csv(
    ED_DIR / "edstays.csv.gz",
    usecols=["subject_id", "stay_id", "gender", "arrival_transport"],
)

print("Loading diagnosis...")
diagnosis = pd.read_csv(
    ED_DIR / "diagnosis.csv.gz",
    usecols=["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"],
)
# Keep only primary diagnosis to maintain one row per stay
diagnosis = diagnosis[diagnosis["seq_num"] == 1].drop(columns=["seq_num"])

print("Loading triage...")
triage = pd.read_csv(
    ED_DIR / "triage.csv.gz",
    usecols=["subject_id", "stay_id", "temperature", "heartrate", "o2sat",
             "sbp", "dbp", "pain", "chiefcomplaint", "acuity"],
)

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
print("Merging...")
merged = edstays.merge(diagnosis, on=["subject_id", "stay_id"], how="left")
merged = merged.merge(triage, on=["subject_id", "stay_id"], how="left")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nMerged shape: {merged.shape}")
print(f"Columns: {list(merged.columns)}")
print(f"\nMissing values:\n{merged.isnull().sum()}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
merged.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
