import numpy as np
import noisereduce as nr
from scipy.signal import butter, filtfilt

def enhance_audio(wav, sr=22050):
    # 1. Noise reduction
    wav = nr.reduce_noise(y=wav, sr=sr, stationary=True, prop_decrease=0.75)
    # 2. Dynamic range compression (simple compression)
    threshold = 0.5
    ratio = 4.0
    wav = np.where(np.abs(wav) > threshold, threshold + (np.abs(wav) - threshold) / ratio * np.sign(wav), wav)
    # 3. Final normalization
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.95
    # 4. High-pass filter (remove DC and low freq noise)
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(80, sr, order=5)
    wav = filtfilt(b, a, wav)
    return wav
