"""
Synthetic fin whale call generator for database creation.

Two call types:
  - Downsweep (~50 Hz component): logarithmic chirp with exponential decay
  - 20 Hz pulse: short tonal call with slight downsweep, characteristic double pulse

Parameter randomization is applied per call to simulate natural biological variability.
A batch generator creates a labelled database of .wav files + a metadata CSV.
"""

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey, hann
import soundfile as sf
import os
import csv
from dataclasses import dataclass, field, asdict
from typing import Optional
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram




@dataclass
class DownsweepParams:
    """Parameters for the ~50 Hz downsweep call."""
    fs: float = 1000.0 
    dur: float = 1.5 # raised from 0.8 to 1.5 s for real calls like 1–2 s
    f0: float = 90.0 # lowered from 110 to 90 Hz
    f1: float = 50.0 # raised from 40 to 50 Hz
    tau: float = 0.6 # raised from 0.25 to 0.6 s for natural decay
    alpha: float = 0.15 # tukey window taper fraction (smooth attack+release)
    harmonics: bool = True
    amplitude: float = 0.95

@dataclass
class PulseParams:
    """Parameters for the 20 Hz pulse call."""
    fs: float = 1000.0
    pulse_dur: float = 1.0
    f0: float = 22.0
    f1: float = 17.0
    inter_pulse_gap: float = 0.22 #it was raised from 0.12 to 0.22 s (doublet visible)
    amplitude: float = 0.95





##########################
# Randomization
##########################
def randomize(value: float, delta: float = 1.0, rel: bool = False) -> float:
    if rel:
        noise = np.random.uniform(-delta, delta) * value
    else:
        noise = np.random.uniform(-delta, delta)
    return value + noise


def randomize_downsweep(p: DownsweepParams, delta: float = 1.0) -> DownsweepParams:
    return DownsweepParams(
        fs=p.fs,
        dur=randomize(p.dur, delta=delta * 0.05, rel=True), # +-5% duration
        f0=randomize(p.f0, delta=delta * 2.0), # +-2 Hz start freq
        f1=randomize(p.f1, delta=delta * 1.0), # +-1 Hz end freq
        tau=randomize(p.tau, delta=delta * 0.03, rel=True), # +-3% decay
        alpha=p.alpha,
        harmonics=p.harmonics,
        amplitude=randomize(p.amplitude, delta=delta * 0.03, rel=True),
    )


def randomize_pulse(p: PulseParams, delta: float = 1.0) -> PulseParams:
    return PulseParams(
        fs=p.fs,
        pulse_dur=randomize(p.pulse_dur, delta=delta * 0.05, rel=True),
        f0=randomize(p.f0, delta=delta * 0.5),
        f1=randomize(p.f1, delta=delta * 0.5),
        inter_pulse_gap=randomize(p.inter_pulse_gap, delta=delta * 0.01),
        amplitude=randomize(p.amplitude,delta=delta * 0.03, rel=True),
    )


# ----------------
# Signal synthesis
# ----------------
def fin_whale_downsweep(p: DownsweepParams) -> np.ndarray:
    n = int(p.fs * p.dur)
    t = np.linspace(0, p.dur, n, endpoint=False)

    # phi=0 instead of random — random phi caused apparent upsweep in STFT
    y = chirp(t, f0=p.f0, f1=p.f1, t1=p.dur, method="logarithmic", phi=0)

    # adding the tukey window (smooth fade-in AND fade-out) + exponential decay
    # replaces the old sharp 2%-sample fade-in which left a hard edge at the end
    win = tukey(n, alpha=p.alpha)
    decay = np.exp(-t / p.tau)
    env = win * decay
    env /= (env.max() + 1e-12)  # keep envelope peak at 1

    y = y * env

    # harmonics with the same envelope
    if p.harmonics:
        y += 0.4 * chirp(t, f0=2*p.f0, f1=2*p.f1, t1=p.dur,
                         method="logarithmic", phi=0) * env
        y += 0.2 * chirp(t, f0=3*p.f0, f1=3*p.f1, t1=p.dur,
                         method="logarithmic", phi=0) * env

    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y.astype(np.float32)




def fin_whale_pulse(p: PulseParams) -> np.ndarray:
    def _single_pulse(pulse_dur: float, f0: float, f1: float, fs: float) -> np.ndarray:
        n = int(fs * pulse_dur)
        t = np.linspace(0, pulse_dur, n, endpoint=False)
        # FIX: phi=0 for consistent sweep direction
        y = chirp(t, f0=f0, f1=f1, t1=pulse_dur, method="linear", phi=0)
        # FIX: Hann window — smooth bell shape, zero at both edges, no clicks
        env = hann(n)
        return (y * env).astype(np.float32)

    pulse1 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs)

    # Second pulse: slightly quieter (natural variation)
    amp2 = np.random.uniform(0.85, 1.0)
    pulse2 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs) * amp2

    gap = np.zeros(int(p.fs * p.inter_pulse_gap), dtype=np.float32)

    y = np.concatenate([pulse1, gap, pulse2]).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y



# ---------------------------------------------------------------------------
# Batch database generator
# ---------------------------------------------------------------------------
def generate_database(
        out_dir: str,
        n_downsweep: int = 100,
        n_pulse: int = 100,
        downsweep_params: Optional[DownsweepParams] = None,
        pulse_params: Optional[PulseParams] = None,
        delta: float = 1.0,
        seed: Optional[int] = 42,
) -> str:
    

    if seed is not None:
        np.random.seed(seed)

    dp = downsweep_params or DownsweepParams()
    pp = pulse_params or PulseParams()

    ds_dir = os.path.join(out_dir, "downsweep")
    pulse_dir = os.path.join(out_dir, "pulse")
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(pulse_dir, exist_ok=True)

    metadata_rows = []

    
    
    ##########################
    print(f"Generating {n_downsweep} downsweep calls...")
    for i in range(n_downsweep):
        rp = randomize_downsweep(dp, delta=delta)
        audio = fin_whale_downsweep(rp)
        fname = f"downsweep_{i:04d}.wav"
        sf.write(os.path.join(ds_dir, fname), audio, int(rp.fs))
        metadata_rows.append({
            "filename": os.path.join("downsweep", fname),
            "label": "downsweep",
            "fs": rp.fs,
            "dur": round(rp.dur, 3),
            "f0": round(rp.f0, 3),
            "f1": round(rp.f1, 3),
            "tau": round(rp.tau, 4),
            "harmonics": rp.harmonics,
            "amplitude": round(rp.amplitude, 4),
            "inter_pulse_gap": "",
        })

    
    
    ##########################
    print(f"Generating {n_pulse} pulse calls...")
    for i in range(n_pulse):
        rp = randomize_pulse(pp, delta=delta)
        audio = fin_whale_pulse(rp)
        fname = f"pulse_{i:04d}.wav"
        sf.write(os.path.join(pulse_dir, fname), audio, int(rp.fs))
        metadata_rows.append({
            "filename": os.path.join("pulse", fname),
            "label": "pulse",
            "fs": rp.fs,
            "dur": round(2 * rp.pulse_dur + rp.inter_pulse_gap, 4),
            "f0": round(rp.f0, 3),
            "f1": round(rp.f1, 3),
            "tau": "",
            "harmonics": "",
            "amplitude": round(rp.amplitude, 4),
            "inter_pulse_gap": round(rp.inter_pulse_gap, 4),
        })


    # save
    csv_path = os.path.join(out_dir, "metadata.csv")
    fieldnames = ["filename", "label", "fs", "dur", "f0", "f1",
                  "tau", "harmonics", "amplitude", "inter_pulse_gap"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)



    print(f"\nDatabase complete!")
    print(f"Downsweep calls: {n_downsweep} -> {ds_dir}")
    print(f"Pulse calls: {n_pulse} ->{pulse_dir}")
    print(f"Metadata CSV: {csv_path}")
    return csv_path


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
    plt.show()







def main():
    np.random.seed(0)

    ##########################
    # Downsweep
    dp = DownsweepParams()
    rp_ds = randomize_downsweep(dp)
    ds_audio = fin_whale_downsweep(rp_ds)
    print(f"Downsweep: f0={rp_ds.f0:.1f} Hz -> f1={rp_ds.f1:.1f} Hz, "
          f"dur={rp_ds.dur:.3f}s, tau={rp_ds.tau:.3f}")
    plot_spec(ds_audio, int(rp_ds.fs), "Fin Whale Downsweep for 50 Hz component")


    ##########################
    #20 Hz pulse preview
    pp = PulseParams()
    rp_p = randomize_pulse(pp)
    pulse_audio = fin_whale_pulse(rp_p)
    print(f"Pulse: f0={rp_p.f0:.1f} Hz -> f1={rp_p.f1:.1f} Hz, "
          f"pulse_dur={rp_p.pulse_dur:.3f}s, gap={rp_p.inter_pulse_gap:.3f}s")
    plot_spec(pulse_audio, int(rp_p.fs), "Fin Whale 20 Hz Pulse (doublet)")



    ##########################
    #test database
    generate_database(
        out_dir="fin_whale_db",
        n_downsweep=10,
        n_pulse=10,
        delta=1.0,
        seed=42,
    )


if __name__ == "__main__":
    main()