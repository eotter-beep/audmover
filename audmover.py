import numpy as np
import soundfile as sf
import subprocess
import librosa

path = input("Enter path here: ").strip()

# Load stereo audio
data, samplerate = sf.read(path)

# Invert right channel and mix
if data.ndim == 2 and data.shape[1] == 2:
    data[:,1] = -data[:,1]
    out = data[:,0] + data[:,1]
else:
    raise ValueError("Input must be stereo audio (2 channels).")

sf.write("center_removed.wav", out, samplerate)

# Separate using spleeter
subprocess.run(["spleeter", "separate", "-i", path, "-p", "spleeter:2stems", "-o", "out_spleeter"])

# Apply a simple spectral gate
inst, sr = librosa.load("center_removed.wav", sr=None, mono=True)
S = librosa.stft(inst, n_fft=2048, hop_length=512)
mag, phase = np.abs(S), np.angle(S)
th = np.median(mag) * 0.5
mag[mag < th] = 0
S2 = mag * np.exp(1j * phase)
clean = librosa.istft(S2, hop_length=512)
name = sf.write("vocalless.wav", clean, sr)

print(f"Done! File saved as vocalless.wav")
