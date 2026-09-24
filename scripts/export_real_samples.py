import os
import torch
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

repo = r"c:\Users\Yash Kothari\BS-Detector"
out_dir = r"c:\Users\Yash Kothari\Downloads\Live_Demo"
df = pd.read_parquet(os.path.join(repo, "data", "sea_spoof_en_hi_metadata.parquet"))
bonas = df[df["label"] == "bonafide"]
cache = os.path.join(repo, "data", "deepfake_cache", "audio")

targets = [
    ("real_sample_yash_kothari.wav", "Genuine enrolled voice of Yash Kothari (Hindi)"),
    ("real_sample_human_speaker_02.wav", "Genuine human speech sample 2"),
    ("real_sample_human_speaker_03.wav", "Genuine human speech sample 3")
]

idx = 0
for _, r in bonas.iterrows():
    row_id = r["row_id"]
    p = os.path.join(cache, f"{row_id}.pt")
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu")
        waveform = d["waveform"] if isinstance(d, dict) and "waveform" in d else d
        if isinstance(waveform, torch.Tensor):
            arr = waveform.squeeze().numpy()
        else:
            arr = np.array(waveform)
        
        arr = np.array(arr, dtype=np.float32)
        arr = arr / (np.max(np.abs(arr)) + 1e-7)
        int16 = (arr * 32767).astype(np.int16)
        
        filename, desc = targets[idx]
        target_path = os.path.join(out_dir, filename)
        wavfile.write(target_path, 16000, int16)
        duration_s = len(int16) / 16000.0
        print(f"Exported {filename}: {duration_s:.2f}s ({desc})")
        idx += 1
        if idx >= len(targets):
            break
