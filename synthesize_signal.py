from class_params import DownsweepParams, PulseParams
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey, hann
from randomizer import randomize_downsweep, randomize_pulse
from plots import plot_spec





########################
# DOWNSWEEP
#######################
def fin_whale_downsweep(p: DownsweepParams) -> np.ndarray:
    # logarithmic sweep from f0 to f1 over dur seconds, sampled at fs
    n = int(p.fs * p.dur)
    t = np.linspace(0, p.dur, n, endpoint=False)

    # phi=0 instead of random — random phi caused apparent upsweep in STFT
    y = chirp(t, f0=p.f0, f1=p.f1, t1=p.dur, method="logarithmic", phi=0)

    # adding the tukey window (smooth fade-in AND fade-out) + exponential decay
    # replaces the old sharp 2%-sample fade-in which left a hard edge at the end
    # tau controls the decay rate — higher tau = slower decay, more natural
    win = tukey(n, alpha=p.alpha)
    decay = np.exp(-t / p.tau)
    env = win * decay
    # keep envelope peak at 1
    env /= (env.max() + 1e-12)

    y = y * env

    # harmonics with the same envelope
    if p.harmonics:
        y += 0.4 * chirp(t, f0=2*p.f0, f1=2*p.f1, t1=p.dur,method="logarithmic", phi=0) * env
        y += 0.2 * chirp(t, f0=3*p.f0, f1=3*p.f1, t1=p.dur,method="logarithmic", phi=0) * env

    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y.astype(np.float32)





#####################
# 20 Hz PULSE
#####################
def _single_pulse(pulse_dur: float, f0: float, f1: float, fs: float) -> np.ndarray:
    # linear sweep from f0 to f1 over pulse_dur seconds, sampled at fs
    n = int(fs * pulse_dur)
    t = np.linspace(0, pulse_dur, n, endpoint=False)
    # phi=0 for consistent sweep direction
    y = chirp(t, f0=f0, f1=f1, t1=pulse_dur, method="linear", phi=0)
    # Hann window — smooth bell shape, zero at both edges, no clicks
    env = hann(n)
    return (y * env).astype(np.float32)



def fin_whale_pulse(p: PulseParams) -> np.ndarray:
    pulse1 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs)

    # Second pulse: slightly quieter (natural variation)
    amp2 = np.random.uniform(0.85, 1.0)
    pulse2 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs) * amp2

    gap = np.zeros(int(p.fs * p.inter_pulse_gap), dtype=np.float32)

    y = np.concatenate([pulse1, gap, pulse2]).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y




def _adsr_envelope(n: int, fs: float,attack_s: float,decay_s: float,sustain_level: float,release_s: float) -> np.ndarray:
    """
    Build an ADSR amplitude envelope.

    Shape:
        0 → 1.0  over attack_s      (linear ramp up)
        1.0 → sustain_level  over decay_s    (linear ramp down to sustain)
        sustain_level (flat) for the remaining middle portion
        sustain_level → 0  over release_s   (linear ramp down to 0)

    Args:
        n:             Total number of samples
        fs:            Sample rate (Hz)
        attack_s:      Duration of attack phase (s)
        decay_s:       Duration of decay phase (s)
        sustain_level: Amplitude level during sustain (0–1), e.g. 0.8
        release_s:     Duration of release phase (s)

    Returns:
        env: float32 array of length n, values in [0, 1]
    """
    attack_n  = min(int(attack_s  * fs), n)
    decay_n   = min(int(decay_s   * fs), n - attack_n)
    release_n = min(int(release_s * fs), n)
    sustain_n = max(0, n - attack_n - decay_n - release_n)

    attack  = np.linspace(0.0, 1.0,           attack_n,  endpoint=False)
    decay   = np.linspace(1.0, sustain_level, decay_n,   endpoint=False)
    sustain = np.full(sustain_n, sustain_level)
    release = np.linspace(sustain_level, 0.0, release_n)

    env = np.concatenate([attack, decay, sustain, release])

    # guard: trim or pad to exactly n samples
    if len(env) > n:
        env = env[:n]
    elif len(env) < n:
        env = np.pad(env, (0, n - len(env)))

    return env.astype(np.float32)


def _single_pulse(pulse_dur: float,f0: float, f1: float,fs: float,attack_s: float,decay_s: float,sustain_level: float,release_s: float) -> np.ndarray:
    """
    Args:
        pulse_dur:     Total duration of this pulse (s)
        f0:            Start frequency (Hz)
        f1:            End frequency (Hz)
        fs:            Sample rate (Hz)
        attack_s:      Attack time (s)
        decay_s:       Decay time (s)
        sustain_level: Sustain amplitude (0–1)
        release_s:     Release time (s)
    """
    n = int(fs * pulse_dur)
    t = np.linspace(0, pulse_dur, n, endpoint=False)

    # linear chirp from f0 down to f1
    y = chirp(t, f0=f0, f1=f1, t1=pulse_dur, method="linear", phi=0)

    # ADSR envelope
    env = _adsr_envelope(n, fs,
                         attack_s=attack_s,
                         decay_s=decay_s,
                         sustain_level=sustain_level,
                         release_s=release_s)

    return (y * env).astype(np.float32)




def fin_whale_pulse_doublet(p: PulseParams) -> np.ndarray:
    """
    Each sub-pulse uses an ADSR envelope for realistic amplitude shaping.
    The bandwidth (f0 - f1) and ADSR timings are randomized slightly per call
    via PulseParams (set in randomizer.py).

    Total duration = pulse_dur + inter_pulse_gap + pulse_dur
    """
    pulse1 = _single_pulse(
        pulse_dur=p.pulse_dur,
        f0=p.f0, f1=p.f1,
        fs=p.fs,
        attack_s=p.attack_s,
        decay_s=p.decay_s,
        sustain_level=p.sustain_level,
        release_s=p.release_s,
    )

    # second pulse slightly quieter
    amp2 = np.random.uniform(0.85, 1.0)
    pulse2 = _single_pulse(
        pulse_dur=p.pulse_dur,
        f0=p.f0, f1=p.f1,
        fs=p.fs,
        attack_s=p.attack_s,
        decay_s=p.decay_s,
        sustain_level=p.sustain_level,
        release_s=p.release_s,
    ) * amp2

    gap = np.zeros(int(p.fs * p.inter_pulse_gap), dtype=np.float32)

    y = np.concatenate([pulse1, gap, pulse2]).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y



def fin_whale_pulse(p: PulseParams) -> np.ndarray:
    """
    Just a single pulse with ADSR envelope, no doublet. Useful for testing the pulse shape in isolation.
    """
    y = _single_pulse(
        pulse_dur=p.pulse_dur,
        f0=p.f0, f1=p.f1,
        fs=p.fs,
        attack_s=p.attack_s,
        decay_s=p.decay_s,
        sustain_level=p.sustain_level,
        release_s=p.release_s,
    )
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude

    # plot for testing pruposes
    plot_spec(y, int(p.fs), "Fin Whale 20 Hz Pulse (single)")
    exit()
    return y



########## test
def test_synth(logger=None) -> None:
    """Test the downsweep and pulse synthesis by generating a sample call and plotting its spectrogram"""

    logger.info("TESTING --> Generating a sample downsweep call")
    dp = DownsweepParams()
    rp_ds = randomize_downsweep(dp)
    ds_audio = fin_whale_downsweep(rp_ds)
    logger.info(f"Downsweep: f0={rp_ds.f0:.1f} Hz -> f1={rp_ds.f1:.1f} Hz, "
          f"dur={rp_ds.dur:.3f}s, tau={rp_ds.tau:.3f}")
    plot_spec(ds_audio, int(rp_ds.fs), "Fin Whale Downsweep for 50 Hz component")


    logger.info("TESTING --> Generating a sample 20 Hz pulse call")
    pp = PulseParams()
    rp_p = randomize_pulse(pp)
    pulse_audio = fin_whale_pulse(rp_p)
    logger.info(f"Pulse: f0={rp_p.f0:.1f} Hz -> f1={rp_p.f1:.1f} Hz, "
          f"pulse_dur={rp_p.pulse_dur:.3f}s, gap={rp_p.inter_pulse_gap:.3f}s")
    plot_spec(pulse_audio, int(rp_p.fs), "Fin Whale 20 Hz Pulse (doublet)")
