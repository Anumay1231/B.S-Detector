"""Phase 2: Dense scan for more Hindi bonafide rows, then create corrected subset.

We have 700 Hindi rows from sparse probing (500 spoof, 200 bonafide).
Need 400+400. This script:
  1. Analyzes sparse probe to find bonafide-rich offset regions
  2. Dense-scans those regions for more bonafide rows
  3. Combines all Hindi rows and does stratified sampling
  4. Saves final corrected metadata parquet
"""
import sys, time, json
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Load Phase 1 results
df_phase1 = pd.read_parquet("data/hindi_sparse_probe.parquet")
print(f"Phase 1 rows: {len(df_phase1)}", flush=True)
print(f"  Labels: {dict(df_phase1['label'].value_counts())}", flush=True)

bonafide_offsets = sorted(df_phase1[df_phase1["label"] == "bonafide"]["_offset"].tolist())
spoof_offsets = sorted(df_phase1[df_phase1["label"] == "spoof"]["_offset"].tolist())
print(f"  Bonafide offset range: {min(bonafide_offsets):,} - {max(bonafide_offsets):,}", flush=True)
print(f"  Spoof offset range: {min(spoof_offsets):,} - {max(spoof_offsets):,}", flush=True)

# Find bonafide-rich offset clusters
# Group bonafide offsets into ranges (within 1000 of each other)
clusters = []
current_cluster = [bonafide_offsets[0]]
for off in bonafide_offsets[1:]:
    if off - current_cluster[-1] < 2000:
        current_cluster.append(off)
    else:
        clusters.append((min(current_cluster), max(current_cluster), len(current_cluster)))
        current_cluster = [off]
clusters.append((min(current_cluster), max(current_cluster), len(current_cluster)))

print(f"\nBonafide clusters:", flush=True)
for start, end, count in clusters:
    print(f"  Offset {start:,}-{end:,}: {count} bonafide rows", flush=True)

# Phase 2: Dense scan around bonafide-rich regions
print(f"\n=== Phase 2: Dense scanning for bonafide rows ===", flush=True)
from huggingface_hub import get_token
token = get_token()
import requests

headers = {"Authorization": f"Bearer {token}"}
BASE = "https://datasets-server.huggingface.co"
DATASET = "Jack-ppkdczgx/SEA-Spoof"
BATCH = 100
TOTAL = 439362

# Dense scan: for each bonafide cluster, scan ±5000 rows around it
# Also scan some unexplored regions between clusters
scan_ranges = []
for start, end, count in clusters:
    scan_start = max(0, start - 5000)
    scan_end = min(TOTAL, end + 5000)
    scan_ranges.append((scan_start, scan_end))

# Merge overlapping ranges
scan_ranges.sort()
merged = [scan_ranges[0]]
for s, e in scan_ranges[1:]:
    if s <= merged[-1][1] + 1000:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))

# Also add some unexplored middle regions to catch any missed Hindi bonafide
# The dataset is ~439K rows. Hindi is ~14%. Let's scan a few more areas.
extra_offsets = list(range(50000, 200000, 8787))  # ~17 extra probes
for off in extra_offsets:
    merged.append((off, off + BATCH))

print(f"Scan regions: {len(merged)}", flush=True)
total_to_scan = sum(e - s for s, e in merged)
print(f"Total rows to scan: ~{total_to_scan:,}", flush=True)

t0 = time.time()
new_hindi = []
existing_row_ids = set(df_phase1["row_id"].tolist())
scanned = 0

for region_idx, (reg_start, reg_end) in enumerate(merged):
    offset = reg_start
    while offset < reg_end:
        try:
            r = requests.get(f"{BASE}/rows", params={
                "dataset": DATASET, "config": "default", "split": "train",
                "offset": offset, "length": BATCH,
            }, headers=headers, timeout=30)
            
            if r.status_code != 200:
                offset += BATCH
                continue
            
            rows = r.json().get("rows", [])
            if not rows:
                break
            
            for item in rows:
                row = item.get("row", {})
                if row.get("language") == "hi":
                    rid = row.get("row_id")
                    if rid not in existing_row_ids:
                        entry = {}
                        for k in ["row_id", "utterance_id", "text", "language",
                                   "label", "spoof_type", "category", "split",
                                   "text_source", "mapping_source", "text_granularity",
                                   "is_text_exact", "source_model", "source_dataset",
                                   "speaker_or_voice", "sampling_rate", "audio_was_resampled"]:
                            entry[k] = row.get(k)
                        new_hindi.append(entry)
                        existing_row_ids.add(rid)
            
            scanned += len(rows)
            offset += len(rows)
        except Exception as e:
            offset += BATCH
    
    elapsed = time.time() - t0
    new_bonafide = sum(1 for r in new_hindi if r["label"] == "bonafide")
    new_spoof = sum(1 for r in new_hindi if r["label"] == "spoof")
    if (region_idx + 1) % 5 == 0 or region_idx == len(merged) - 1:
        print(f"  Region {region_idx+1}/{len(merged)}: {scanned:,} scanned, "
              f"+{len(new_hindi)} new Hindi (+{new_bonafide} bona, +{new_spoof} spoof) | "
              f"{elapsed:.0f}s", flush=True)
    
    # Early stop if we have enough
    total_bonafide = len(df_phase1[df_phase1["label"] == "bonafide"]) + new_bonafide
    total_spoof = len(df_phase1[df_phase1["label"] == "spoof"]) + new_spoof
    if total_bonafide >= 500 and total_spoof >= 500:
        print(f"  Sufficient rows collected! ({total_bonafide} bona, {total_spoof} spoof)", flush=True)
        break

elapsed = time.time() - t0
print(f"\nPhase 2 done: {scanned:,} scanned, {len(new_hindi)} new Hindi in {elapsed:.0f}s")

# Combine Phase 1 + Phase 2
df_new = pd.DataFrame(new_hindi) if new_hindi else pd.DataFrame()
if "_offset" not in df_phase1.columns:
    df_phase1["_offset"] = -1
if not df_new.empty and "_offset" not in df_new.columns:
    df_new["_offset"] = -1

if not df_new.empty:
    df_all = pd.concat([df_phase1, df_new], ignore_index=True)
else:
    df_all = df_phase1.copy()

# Deduplicate by row_id
df_all = df_all.drop_duplicates(subset="row_id", keep="first")
print(f"\nCombined pool: {len(df_all)} unique Hindi rows")
print(f"  Labels: {dict(df_all['label'].value_counts())}")
print(f"  Source models: {dict(df_all['source_model'].value_counts())}")

# Save full pool
df_all.to_parquet("data/hindi_full_pool.parquet", index=False)
print(f"Saved full pool to data/hindi_full_pool.parquet")

# Stratified sampling
N_PER_CLASS = 400
spoof_df = df_all[df_all["label"] == "spoof"]
bona_df = df_all[df_all["label"] == "bonafide"]
print(f"\nAvailable: {len(spoof_df)} spoof, {len(bona_df)} bonafide")

if len(spoof_df) < N_PER_CLASS or len(bona_df) < N_PER_CLASS:
    print(f"WARNING: Not enough rows for {N_PER_CLASS} per class!")
    N_PER_CLASS = min(len(spoof_df), len(bona_df))
    print(f"Reduced to {N_PER_CLASS} per class")

rng = np.random.RandomState(42)

# Spoof: stratified by source_model
model_counts = spoof_df["source_model"].value_counts()
print(f"\nSpoof stratified sampling ({len(model_counts)} models):")
spoof_sampled = []
total_in_spoof = len(spoof_df)
allocations = {}
remaining = N_PER_CLASS

for model in model_counts.index:
    alloc = max(1, int(round(N_PER_CLASS * model_counts[model] / total_in_spoof)))
    alloc = min(alloc, model_counts[model])
    allocations[model] = alloc
    remaining -= alloc

while remaining > 0:
    for model in model_counts.index:
        if remaining <= 0:
            break
        if allocations[model] < model_counts[model]:
            allocations[model] += 1
            remaining -= 1

while remaining < 0:
    for model in reversed(model_counts.index):
        if remaining >= 0:
            break
        if allocations[model] > 1:
            allocations[model] -= 1
            remaining += 1

for model, alloc in allocations.items():
    model_df = spoof_df[spoof_df["source_model"] == model]
    sampled = model_df.sample(n=alloc, random_state=rng)
    spoof_sampled.append(sampled)
    print(f"  {model}: {alloc}/{model_counts[model]}")

# Bonafide: simple random sample
bona_sampled = bona_df.sample(n=N_PER_CLASS, random_state=rng)
print(f"\nBonafide: {N_PER_CLASS}/{len(bona_df)}")

# Combine
df_final = pd.concat(spoof_sampled + [bona_sampled], ignore_index=True)

# Drop internal columns
if "_offset" in df_final.columns:
    df_final = df_final.drop(columns=["_offset"])

# Save corrected metadata
output_path = "data/sea_spoof_en_hi_metadata_v2.parquet"
df_final.to_parquet(output_path, index=False)
print(f"\nSaved corrected subset: {len(df_final)} rows to {output_path}")

# Final diagnostics
print(f"\n{'='*60}")
print("CORRECTED SUBSET DIAGNOSTICS")
print(f"{'='*60}")
print(f"Total: {len(df_final)}")
for label in ["spoof", "bonafide"]:
    sub = df_final[df_final["label"] == label]
    print(f"\n--- {label.upper()} ({len(sub)} clips) ---")
    print(f"  source_model:")
    for m, c in sub["source_model"].value_counts().items():
        pct = 100 * c / len(sub)
        print(f"    {m or '(empty)'}: {c} ({pct:.1f}%)")
    print(f"  source_dataset:")
    for d, c in sub["source_dataset"].value_counts().items():
        pct = 100 * c / len(sub)
        print(f"    {d or '(empty)'}: {c} ({pct:.1f}%)")

# Compare with old v1
old_path = "data/sea_spoof_en_hi_metadata.parquet"
df_old = pd.read_parquet(old_path)
overlap = set(df_old["row_id"]) & set(df_final["row_id"])
print(f"\nOverlap with v1: {len(overlap)}/{len(df_old)} row_ids")

print(f"\n{'='*60}")
print("DONE. Next steps:")
print("  1. Delete old caches:")
print("     rmdir /s /q data\\deepfake_cache\\audio")
print("     rmdir /s /q data\\deepfake_cache\\features")
print("  2. Re-run pipeline:")
print("     .\\.venv\\Scripts\\python.exe scripts\\run_deepfake_detection.py \\")
print("       --device cuda --test-per-class 250")
print(f"{'='*60}")
