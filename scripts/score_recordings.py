"""
Score the real-world recordings (docs/recording-protocol.md) with the app's own code.

    python scripts/score_recordings.py [--dir data/real_recordings] [--people people.csv]
                                       [--fakes DIR] [--out report.md]

File names follow the protocol: <person>_<lang>_<setup>_<take>.<ext>, for example
p07_hi_2_b.wav (person p07, Hindi, setup 2, take b). Anything else is skipped and
listed. wav, flac, ogg and mp3 are read directly; m4a and webm need ffmpeg on PATH.

Stage 1 hears what the browser sends: the first 4.04 s of each clip, through the
server's detector head (the first one in HINDI_HEADS). Every real clip should
pass; false alarms are reported by setup. --fakes DIR scores clones made with
scripts/make_fakes.py (of consenting participants only) and counts how many are
caught.

Stage 2 runs the server's speaker check on every same-person pair (same language)
and every different-person pair (same language, same gender when --people gives
genders). Embeddings are computed once per clip with server/speaker.py and the
pair decision uses the same functions as speaker.verify; the first pairs are also
run through speaker.verify itself to make sure the two agree. Pairs are grouped
by language, by setup (both clips in the same setup, or mixed) and by speech
length, in the format of section 5 of docs/speaker-verification-hindi-english.md.

Recordings stay in data/real_recordings/ (gitignored). Never commit them.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "src"))

NAME = re.compile(r"^(?P<person>[A-Za-z0-9]+)_(?P<lang>[a-z]+)_(?P<setup>[1-4])_(?P<take>[a-z])\.(?P<ext>wav|flac|ogg|mp3|m4a|webm)$", re.I)
SETUPS = {"1": "laptop, quiet", "2": "laptop, noisy", "3": "phone, quiet", "4": "phone, noisy"}
LANGS = {"hi": "Hindi", "en": "English"}
AUDIO = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
CHECKED_PAIRS = 5  # pairs also run through speaker.verify as a cross-check


def decode(path: Path) -> np.ndarray:
    import speaker

    if path.suffix.lower() in {".m4a", ".webm"}:
        if not shutil.which("ffmpeg"):
            raise ValueError("needs ffmpeg on PATH (or convert it to .wav)")
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                             capture_output=True, check=True).stdout
        return speaker.read_wav(out)
    return speaker.read_wav(path.read_bytes())


def load_people(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["person"].strip(): r.get("gender", "").strip().lower() for r in rows}


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "real_recordings")
    ap.add_argument("--people", type=Path, help="people.csv with person and gender columns")
    ap.add_argument("--fakes", type=Path, help="folder of cloned voices to score with stage 1")
    ap.add_argument("--out", type=Path, help="also write the report to this file")
    a = ap.parse_args()
    lines: list[str] = []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731

    clips, skipped = [], []
    for p in sorted(a.dir.iterdir()) if a.dir.is_dir() else []:
        if p.suffix.lower() not in AUDIO:
            continue
        m = NAME.match(p.name)
        (clips.append({"path": p, **m.groupdict()}) if m else skipped.append(p.name))
    if not clips:
        raise SystemExit(f"No recordings named like p07_hi_2_b.wav in {a.dir}.")
    people = load_people(a.people)

    import app
    import speaker
    import torch
    from app import engine

    print("Loading the detector and the speaker encoders ...", flush=True)
    app.model.load()
    app.voices.load()
    if app.model.state != "ready" or app.voices.state != "ready":
        raise SystemExit(f"Models failed to load: {app.model.message} / {app.voices.message}")
    state = app.voices

    def spoof_prob(x: np.ndarray) -> float:
        return float(app.detect(speaker.wav_bytes(x[: engine.INPUT_LENGTH]), full_clip=False).prob_spoof)

    say(f"# Real-recording results ({len(clips)} clips in {a.dir.name}/)")
    say("")
    say(f"Detector head: {app.model.head_kind}. Speaker check: {state.as_json()['model']} ({state.system}).")
    if skipped:
        say(f"Skipped {len(skipped)} file(s) not named <person>_<lang>_<setup>_<take>: {', '.join(skipped[:10])}"
            + (" ..." if len(skipped) > 10 else ""))

    # --- per clip: decode, stage 1, speech, embeddings ---------------------------
    refused: list[str] = []
    for c in clips:
        try:
            audio = decode(c["path"])
        except Exception as exc:  # noqa: BLE001
            refused.append(f"{c['path'].name}: can't read it ({exc})")
            c["audio"] = None
            continue
        c["audio"] = audio
        c["p_spoof"] = spoof_prob(audio)
        c["usable"] = False
        if len(audio) < speaker.MIN_SECONDS * speaker.SAMPLE_RATE or speaker.speech_snr(audio) < speaker.MIN_SNR_DB:
            refused.append(f"{c['path'].name}: too short or too noisy for the speaker check")
            continue
        speech = speaker.speech_only(audio)
        if len(speech) < speaker.MIN_SPEECH_SECONDS * speaker.SAMPLE_RATE:
            refused.append(f"{c['path'].name}: only {len(speech) / speaker.SAMPLE_RATE:.1f} s of clear speech")
            continue
        c["usable"] = True
        c["speech_s"] = len(speech) / speaker.SAMPLE_RATE
        with torch.no_grad():
            c["ecapa"] = speaker.embed(state.encoder, speech)
            c["wespeaker"] = speaker.embed_second(state.second, speech) if state.system == "fused" else None
            c["whole"] = state.encoder.encode(torch.from_numpy(np.ascontiguousarray(speech)).unsqueeze(0))

    # --- stage 1 -------------------------------------------------------------------
    say("")
    say("## Deepfake check on real clips (first 4.04 s, as the browser sends it)")
    say("")
    say("| Language | Setup | Real clips | Wrongly flagged | False-alarm rate |")
    say("|---|---|---|---|---|")
    by_setup: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in clips:
        if c["audio"] is not None:
            by_setup[(c["lang"].lower(), c["setup"])].append(c["p_spoof"])
    for (lang, setup), ps in sorted(by_setup.items()):
        flagged = sum(p >= 0.5 for p in ps)
        say(f"| {LANGS.get(lang, lang)} | {setup} ({SETUPS[setup]}) | {len(ps)} | {flagged} | {pct(flagged, len(ps))} |")
    flagged_files = [c for c in clips if c["audio"] is not None and c["p_spoof"] >= 0.5]
    if flagged_files:
        say("")
        say("Flagged: " + ", ".join(f"{c['path'].name} ({c['p_spoof']:.2f})" for c in flagged_files))

    if a.fakes:
        fakes = [p for p in sorted(a.fakes.iterdir()) if p.suffix.lower() in AUDIO]
        caught = []
        for p in fakes:
            try:
                caught.append(spoof_prob(decode(p)) >= 0.5)
            except Exception as exc:  # noqa: BLE001
                refused.append(f"{p.name}: can't read it ({exc})")
        say("")
        say(f"Clones in {a.fakes.name}/: {sum(caught)} of {len(caught)} caught ({pct(sum(caught), len(caught))}).")

    # --- stage 2 -------------------------------------------------------------------
    def decide(x: dict, y: dict) -> dict:
        cos = {"ecapa": speaker._cos(x["ecapa"], y["ecapa"])}
        if state.system == "fused":
            cos["wespeaker"] = speaker._cos(x["wespeaker"], y["wespeaker"])
        score = speaker.fused_score(cos) if state.system == "fused" else cos["ecapa"]
        d = min(x["speech_s"], y["speech_s"])
        threshold, margin = speaker.threshold_for(d, state.system), speaker.margin_for(d, state.system)
        return {
            "d": d,
            "old": speaker._cos(x["whole"], y["whole"]) >= speaker.VOXCELEB_THRESHOLD,
            "ecapa_dur": cos["ecapa"] >= speaker.threshold_for(d, "ecapa"),
            "fused": score >= threshold,
            "outcome": speaker.decide(score, threshold, margin),
            "score": score,
        }

    usable = [c for c in clips if c.get("usable")]
    rows: dict[tuple[str, str, str], list[tuple[bool, dict]]] = defaultdict(list)
    checked = 0
    for x, y in itertools.combinations(usable, 2):
        if x["lang"].lower() != y["lang"].lower():
            continue
        same = x["person"] == y["person"]
        if not same and people and people.get(x["person"]) != people.get(y["person"]):
            continue  # impostor pairs are same-gender when genders are known
        r = decide(x, y)
        if checked < CHECKED_PAIRS:
            full = speaker.verify(state, x["audio"], y["audio"])
            if full["outcome"] != r["outcome"] or abs(full["score"] - r["score"]) > 1e-3:
                raise SystemExit(f"Pair scoring disagrees with speaker.verify on {x['path'].name} vs {y['path'].name}: {full['score']} vs {r['score']}")
            checked += 1
        setup = f"setup {x['setup']}" if x["setup"] == y["setup"] else "mixed setups"
        bucket = "3-8 s" if r["d"] < speaker.RELIABLE_SPEECH_SECONDS else "8 s+"
        rows[(LANGS.get(x["lang"].lower(), x["lang"]), setup, bucket)].append((same, r))

    say("")
    say("## Speaker check")
    say("")
    say("Balanced accuracy = mean of (genuine accepted, impostor rejected). \"Setup N\" means both clips were")
    say("recorded in setup N; \"mixed setups\" pairs clips from different setups. \"Old\" is whole-clip cosine")
    say(f"at the VoxCeleb cut-off {speaker.VOXCELEB_THRESHOLD:.4f}, with the encoder the app loads now.")
    say("")
    say("| Set | Condition | Speech | Old: impostors accepted | Old accuracy | ECAPA + duration threshold | Fused | **Fused + unsure: accuracy on answers** | Answered | Pairs (gen / imp) |")
    say("|---|---|---|---|---|---|---|---|---|---|")

    def bal(pairs: list[tuple[bool, dict]], key: str) -> str:
        gen = [r[key] for s, r in pairs if s]
        imp = [not r[key] for s, r in pairs if not s]
        parts = [np.mean(v) for v in (gen, imp) if v]
        return f"{100 * np.mean(parts):.1f}%" if parts else "-"

    for (lang, setup, bucket), pairs in sorted(rows.items()):
        gen = [r for s, r in pairs if s]
        imp = [r for s, r in pairs if not s]
        old_fa = pct(sum(r["old"] for r in imp), len(imp))
        answered = [(s, r) for s, r in pairs if r["outcome"] != "UNCERTAIN"]
        on_answers = bal([(s, {"m": r["outcome"] == "MATCH"}) for s, r in answered], "m")
        say(f"| {lang} | {setup} | {bucket} | {old_fa} | {bal(pairs, 'old')} | {bal(pairs, 'ecapa_dur')} | {bal(pairs, 'fused')} "
            f"| **{on_answers}** | {pct(len(answered), len(pairs))} | {len(gen)} / {len(imp)} |")
    if not rows:
        say("| - | - | - | - | - | - | - | - | - | no usable pairs |")

    if refused:
        say("")
        say(f"Not used ({len(refused)}):")
        for r in refused:
            say(f"- {r}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
