from dataclasses import dataclass


@dataclass
class DownsweepParams:
    """Parameters for the 50 Hz downsweep call"""
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
    """Parameters for the 20 Hz pulse call"""
    fs: float = 1000.0
    pulse_dur: float = 1.0
    f0: float = 22.0
    f1: float = 17.0
    inter_pulse_gap: float = 0.22 #it was raised from 0.12 to 0.22 s (doublet visible)
    amplitude: float = 0.95


@dataclass
class InjectionEvent:
    """Metadata for a single call injection event"""
    call_type: str # "downsweep" or "pulse"
    offset_s: float # start time in the mixed recording (seconds)
    duration_s: float # duration of the call (seconds)
    snr_db: float # target SNR (dB) used for scaling
    actual_snr_db: float # SNR measured after injection (sanity check)
    scale_factor: float # amplitude scale applied to the call