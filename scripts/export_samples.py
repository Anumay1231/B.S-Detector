import os
import torch
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

repo_dir = r"c:\Users\Yash Kothari\BS-Detector"
out_dir = r"c:\Users\Yash Kothari\Downloads\Live_Demo"
parquet_path = os.path.join(repo_dir, "data", "sea_spoof_en_hi_metadata.parquet")
audio_cache_dir = os.path.join(repo_dir, "data", "deepfake_cache", "audio")

df = pd.read_parquet(parquet_path)
spoofs = df[df["label"] == "spoof"]
bonafides = df[df["label"] == "bonafide"]

def extract_audio_array(data):
    if isinstance(data, dict):
        if "array" in data:
            arr = data["array"]
        elif "audio" in data:
            arr = data["audio"]
        else:
            # first value that is array/tensor
            for k, v in data.items():
                if isinstance(v, (np.ndarray, torch.Tensor, list)):
                    arr = v
                    break
    elif isinstance(data, torch.Tensor):
        arr = data.squeeze().numpy()
    else:
        arr = np.array(data)

    if isinstance(arr, torch.Tensor):
        arr = arr.squeeze().numpy()
    arr = np.array(arr, dtype=np.float32)
    return arr

# 1. Export benchmark spoof
for _, row in spoofs.iterrows():
    row_id = row["row_id"]
    pt_path = os.path.join(audio_cache_dir, f"{row_id}.pt")
    if os.path.exists(pt_path):
        data = torch.load(pt_path, map_location="cpu")
        arr = extract_audio_array(data)
        arr = arr / (np.max(np.abs(arr)) + 1e-7)
        int16_arr = (arr * 32767).astype(np.int16)
        
        out_path = os.path.join(out_dir, "fake_sample_deepfake_hindi.wav")
        wavfile.write(out_path, 16000, int16_arr)
        print(f"Exported benchmark deepfake to: {out_path} ({len(int16_arr)/16000:.2f}s, model: {row.get('source_model')})")
        break

# 2. Export benchmark bonafide
for _, row in bonafides.iterrows():
    row_id = row["row_id"]
    pt_path = os.path.join(audio_cache_dir, f"{row_id}.pt")
    if os.path.exists(pt_path):
        data = torch.load(pt_path, map_location="cpu")
        arr = extract_audio_array(data)
        arr = arr / (np.max(np.abs(arr)) + 1e-7)
        int16_arr = (arr * 32767).astype(np.int16)
        
        out_path = os.path.join(out_dir, "authentic_sample_hindi.wav")
        wavfile.write(out_path, 16000, int16_arr)
        print(f"Exported benchmark bonafide to: {out_path} ({len(int16_arr)/16000:.2f}s)")
        break

# 3. Generate high-tech synthetic vocoder attack sample
sr = 16000
duration = 4.2
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
f0 = 145 + 18 * np.sin(2 * np.pi * 3.2 * t)
carrier = np.sin(2 * np.pi * f0 * t) + 0.45 * np.sin(2 * np.pi * 2 * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
envelope = np.clip(np.sin(2 * np.pi * 1.1 * t) * np.sin(2 * np.pi * 0.45 * t), 0, 1)
noise_jitter = (np.random.rand(len(t)) - 0.5) * 0.07 * np.sin(2 * np.pi * 50 * t)
vocoder_spike = 0.05 * np.sin(2 * np.pi * 4600 * t)

synth_spoof = (carrier * envelope + noise_jitter + vocoder_spike)
synth_spoof = synth_spoof / (np.max(np.abs(synth_spoof)) + 1e-7)
synth_int16 = (synth_spoof * 32767).astype(np.int16)

out_synth = os.path.join(out_dir, "fake_sample_ai_clone_attack.wav")
wavfile.write(out_synth, sr, synth_int16)
print(f"Exported synthesized AI clone attack to: {out_synth}")
