from datasets import load_dataset
from collections import Counter

DATASET = "Jack-ppkdczgx/SEA-Spoof"
N = 5000

print("Loading SEA-Spoof in streaming mode...")

ds = load_dataset(
    DATASET,
    split="train",
    streaming=True,
)

print("\nDataset:")
print(ds)

# IMPORTANT:
# Force the streaming dataset to return only metadata fields.
metadata_columns = [
    "row_id",
    "utterance_id",
    "text",
    "language",
    "label",
    "spoof_type",
    "category",
    "split",
    "text_source",
    "mapping_source",
    "text_granularity",
    "is_text_exact",
    "source_model",
    "source_dataset",
    "speaker_or_voice",
    "sampling_rate",
    "audio_was_resampled",
]

ds = ds.select_columns(metadata_columns)

print(f"\nAuditing first {N} training samples...")

language_counts = Counter()
label_counts = Counter()
spoof_counts = Counter()
source_model_counts = Counter()

first_rows = []

for i, sample in enumerate(ds):
    if i < 20:
        first_rows.append(sample)

    language_counts[sample["language"]] += 1
    label_counts[sample["label"]] += 1
    spoof_counts[sample["spoof_type"]] += 1
    source_model_counts[sample["source_model"]] += 1

    if i + 1 >= N:
        break

print("\n" + "=" * 70)
print("FIRST 20 SAMPLES")
print("=" * 70)

for i, row in enumerate(first_rows):
    print(
        f"{i+1:03d} | "
        f"lang={row['language']} | "
        f"label={row['label']} | "
        f"spoof={row['spoof_type']} | "
        f"source={row['source_model']}"
    )

print("\n" + "=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)

for k, v in language_counts.most_common():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

for k, v in label_counts.most_common():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("SPOOF TYPE DISTRIBUTION")
print("=" * 70)

for k, v in spoof_counts.most_common():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("SOURCE MODEL DISTRIBUTION")
print("=" * 70)

for k, v in source_model_counts.most_common():
    print(f"{k}: {v}")

print("\nAudit complete.")