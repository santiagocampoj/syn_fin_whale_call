import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram
import numpy as np
import soundfile as sf



def plot_spec(audio, fs, title, save=False, pad_s=1):
    # pad 0.5 is added to show the real ocean noise before and after the call in the spectrogram
    pad = np.zeros(int(fs * pad_s), dtype=np.float32)
    audio_padded = np.concatenate([pad, audio, pad])

    f, t, Sxx = scipy_spectrogram(audio_padded, fs=fs, nperseg=512, noverlap=480, window="hann", scaling="spectrum")
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="inferno", vmin=-110, vmax=-20)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.ylim(0, 100)

    # mark where the actual call starts and ends with vertical lines
    plt.axvline(x=pad_s, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(x=pad_s + len(audio)/fs, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    if save:
        plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    else:
        plt.show()




def plot_spec_file(path, title=None, save=False, pad_s=1, logger=None):
    audio, fs = sf.read(path, dtype="float32", always_2d=False)
    logger.info(f"Loaded: {len(audio)} samples, fs={fs} Hz, dur={len(audio)/fs:.2f}s")
    plot_spec(audio, fs, title or path, save=save, pad_s=pad_s)