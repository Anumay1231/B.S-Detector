"""
Phase 12 data loaders: real speech (SVQ, Common Voice) and IndicSynth fakes, with
each clip's split (train / val / test) decided by person. See phase12_common.py.
"""

from __future__ import annotations

import glob
import io
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from phase12_common import DATA, DEMO_CV_CLIPS, DEMO_SVQ, rank_split, split_of

HUB = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")


def _files(pattern: str) -> list[str]:
    return sorted(glob.glob(str(HUB / pattern)))


def svq() -> list[dict]:
    """Every SVQ hi_in/en_in utterance, all four environments, minus the demo speakers.
    SVQ has about 20 people, so men and women are split separately."""
    rows = []
    for f in _files("datasets--google--svq/snapshots/*/2.0.0/utts_*_in_*.parquet"):
        t = pq.read_table(f, columns=["speaker_id", "speaker_gender", "environment", "locale", "waveform"])
        cols = [t.column(c).to_pylist() for c in ("speaker_id", "speaker_gender", "environment", "locale", "waveform")]
        for spk, gender, env, loc, w in zip(*cols):
            if spk in DEMO_SVQ:
                continue
            rows.append({"source": f"svq-{env}", "lang": loc[:2], "speaker": f"svq:{spk}", "gender": gender, "audio": w["bytes"]})
    split: dict[str, str] = {}
    by_gender = defaultdict(set)
    for r in rows:
        by_gender[r["gender"] if r["gender"] in ("male", "female") else "other"].add(r["speaker"])
    for keys in by_gender.values():
        split.update(rank_split(keys))
    for r in rows:
        r["split"] = split[r["speaker"]]
    return rows


def common_voice(lang: str, with_audio: bool = True) -> list[dict]:
    """Common Voice 17: Hindi validated, English first 8 train shards, minus demo speakers."""
    pattern = "hi/validated/*/*.parquet" if lang == "hi" else "en/train-*.parquet"
    files = _files(f"datasets--fixie-ai--common_voice_17_0/snapshots/*/{pattern}")
    if lang == "en":
        files = files[:8]
    cols = ["client_id", "path", "sentence", "gender", "accent"] + (["audio"] if with_audio else [])
    rows = []
    for f in files:
        t = pq.read_table(f, columns=cols)
        rows += [dict(zip(cols, v)) for v in zip(*(t.column(c).to_pylist() for c in cols))]
    demo = {r["client_id"] for r in rows if r["path"] in DEMO_CV_CLIPS}
    out = []
    for r in rows:
        if r["client_id"] in demo:
            continue
        spk = f"cv-{lang}:{r['client_id']}"
        # Indian-accented English is its own source, so balanced sampling gives it equal weight
        indian = lang == "en" and "india" in (r.get("accent") or "").lower()
        item = {"source": "cv-en-in" if indian else f"cv-{lang}", "lang": lang, "speaker": spk, "split": split_of(spk),
                "gender": r.get("gender") or "", "sentence": r["sentence"], "path": r["path"]}
        if with_audio:
            item["audio"] = r["audio"]["bytes"]
            try:
                item["seconds"] = sf.info(io.BytesIO(item["audio"])).duration
            except Exception:  # noqa: BLE001
                item["seconds"] = 0.0
        out.append(item)
    return out


def indicsynth() -> list[dict]:
    """IndicSynth Hindi (XTTS-v2 and FreeVC24) from the downloaded shards, split by speaker."""
    rows = []
    for f in _files("datasets--vdivyasharma--IndicSynth/snapshots/*/Hindi/*.parquet"):
        # the target speaker is the voice the fake imitates; split by that
        cols = ("audio", "Target Speaker ID", "Gender", "Generative Model")
        t = pq.read_table(f, columns=list(cols))
        for a, spk, g, m in zip(*(t.column(c).to_pylist() for c in cols)):
            rows.append({"source": f"indicsynth-{m}", "lang": "hi", "speaker": f"indicsynth:{spk}", "gender": (g or "").lower(),
                         "audio": a["bytes"]})
    split = rank_split(r["speaker"] for r in rows)
    for r in rows:
        r["split"] = split[r["speaker"]]
    return rows


def fleurs() -> list[dict]:
    """FLEURS Hindi read speech. It has no speaker ids, so its own train/test split is kept;
    the validation split holds the bundled real-hindi sample and is not used."""
    rows = []
    for split in ("train", "test"):
        for f in _files(f"datasets--google--fleurs/snapshots/*/hi_in/{split}/0000.parquet"):
            t = pq.read_table(f, columns=["audio", "gender"])
            for a, g in zip(t.column("audio").to_pylist(), t.column("gender").to_pylist()):
                rows.append({"source": "fleurs-hi", "lang": "hi", "speaker": f"fleurs:{split}:{len(rows) // 50}",
                             "gender": str(g), "split": split, "audio": a["bytes"]})
    return rows


def indictts() -> list[dict]:
    """IndicTTS Hindi: two professional speakers recorded in a studio. Training only: the
    head must learn that a clean studio voice is not by itself a sign of a fake."""
    rows = []
    for f in _files("datasets--SPRINGLab--IndicTTS-Hindi/snapshots/*/data/*.parquet"):
        t = pq.read_table(f, columns=["audio", "gender"])
        for a, g in zip(t.column("audio").to_pylist(), t.column("gender").to_pylist()):
            rows.append({"source": "indictts-hi", "lang": "hi", "speaker": f"indictts:{g}", "gender": str(g),
                         "split": "train", "audio": a["bytes"]})
    return rows


GATED = DATA / "gated"  # parquet files fetched by hand (they need a Hugging Face login)
PROFILE = ("gender", "age-group", "primary_language", "native_place_state", "native_place_district",
           "highest_qualification", "job_category", "occupation_domain")


def ai4bharat(name: str, lang: str) -> list[dict]:
    """Svarah (Indian English) or Lahaja (Hindi accents) from data/raw/phase12/gated/<name>/.
    Neither has speaker ids, so a speaker is the full profile (gender, age, mother tongue,
    district, education, job), which is close to unique per person."""
    rows = []
    for f in sorted((GATED / name).glob("*.parquet")):
        p = pq.ParquetFile(f)
        cols = [c for c in PROFILE if c in p.schema_arrow.names]
        speaker_col = next((c for c in ("speaker_id", "sp_id", "speaker") if c in p.schema_arrow.names), None)
        t = p.read(columns=["audio_filepath"] + cols + ([speaker_col] if speaker_col else []))
        audio = t.column("audio_filepath").to_pylist()
        meta = [t.column(c).to_pylist() for c in cols]
        spk = t.column(speaker_col).to_pylist() if speaker_col else None
        for i, a in enumerate(audio):
            key = str(spk[i]) if spk else "|".join(str(m[i]) for m in meta)
            rows.append({"source": name, "lang": lang, "speaker": f"{name}:{key}",
                         "gender": str(meta[0][i]).lower() if cols else "", "audio": a["bytes"]})
    split: dict[str, str] = {}
    by_gender = defaultdict(set)
    for r in rows:
        by_gender[r["gender"]].add(r["speaker"])
    for keys in by_gender.values():
        split.update(rank_split(keys))
    for r in rows:
        r["split"] = split[r["speaker"]]
    return rows
