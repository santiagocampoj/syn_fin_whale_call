import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram
import numpy as np



def plot_spec(audio, fs, title):
    # nperseg=512, noverlap=480 for much better frequency resolution
    f, t, Sxx = scipy_spectrogram(audio, fs=fs, nperseg=512, noverlap=480,
                                   window="hann", scaling="spectrum")
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud",
                   cmap="inferno", vmin=-110, vmax=-20)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.ylim(0, 200)
    plt.tight_layout()
    # plt.show()

    # save
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
