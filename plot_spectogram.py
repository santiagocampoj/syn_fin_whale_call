import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram




def load_audio(path: str, mono: bool = True):
    try:
        import soundfile as sf
        y, fs = sf.read(path, always_2d=True)
        if mono:
            y = y.mean(axis=1)
        else:
            y = y.T  # (C, N)
        y = y.astype(np.float32)
        return y, float(fs)
    except Exception:
        pass

    except ImportError as e:
        raise SystemExit("Could not read this file.\n") from e




def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

def auto_stft_params(fs: float, n_samples: int, win_sec: float = 0.05, overlap_ratio: float = 0.75):
    # targer window length in samples
    target = int(round(fs * win_sec)) # target window length in samples

    # nperseg is the next power of 2 of target, at least 64 samples
    nperseg = next_pow2(max(64, target)) # at least 64 samples, next power of 2 because FFT efficiency
    # making sure the nperseg is not longer than the signal
    nperseg = min(nperseg, max(8, n_samples))

    # noverlap
    noverlap = int(round(overlap_ratio * nperseg))
    # making sure noverlap < nperseg
    noverlap = min(noverlap, nperseg - 1)
    return nperseg, noverlap



def plot_spec(y: np.ndarray, fs: float, nperseg: int, noverlap: int, fmax=None):
    # if stereo, average it
    if y.ndim == 2:
        y = y.mean(axis=0)

    #getting spectrogram parameters
    f, t, Sxx = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap)


    #plotting
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    plt.colorbar()
    plt.ylim(0, (fs / 2) if fmax is None else fmax)
    plt.tight_layout()
    # plt.show()

    #save expectogram
    plt.savefig("spectrogram.png")
    



def argument_parser():
    p = argparse.ArgumentParser(description="Plot a spectrogram from audio file.")
    p.add_argument("-f", "--file", required=True, help="Audio file path")
    p.add_argument("--win-sec", type=float, default=0.05, help="Target window length in seconds (auto nperseg)")
    p.add_argument("--overlap", type=float, default=0.75, help="Overlap ratio in [0,1)")
    p.add_argument("--fmax", type=float, default=None, help="Max freq to display (Hz)")
    p.add_argument("--keep-stereo", action="store_true", help="Keep stereo (still averaged for plotting)")
    args = p.parse_args()
    return args



def main():
    args = argument_parser()

    #reading path
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")


    # loadding audio file
    y, fs = load_audio(str(path), mono=not args.keep_stereo)
    nperseg, noverlap = auto_stft_params(fs, (y.shape[-1] if y.ndim == 2 else len(y)),win_sec=args.win_sec,overlap_ratio=args.overlap)
    plot_spec(y, fs, nperseg, noverlap, fmax=args.fmax)



if __name__ == "__main__":
    main()
