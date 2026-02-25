from typing import Optional
import os 
import csv
import numpy as np
import soundfile as sf
from class_params import DownsweepParams, PulseParams
from randomizer import randomize_downsweep, randomize_pulse
from synthesize_signal import fin_whale_downsweep, fin_whale_pulse



def generate_database(out_dir: str,n_downsweep: int = 100,n_pulse: int = 100,downsweep_params: Optional[DownsweepParams] = None,pulse_params: Optional[PulseParams] = None,delta: float = 1.0,seed: Optional[int] = 42,logger=None) -> str:
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
    logger.info(f"Generating {n_downsweep} downsweep calls...")
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
    logger.info(f"Generating {n_pulse} pulse calls...")
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

    ##########################
    # save
    csv_path = os.path.join(out_dir, "metadata.csv")
    fieldnames = ["filename", "label", "fs", "dur", "f0", "f1",
                  "tau", "harmonics", "amplitude", "inter_pulse_gap"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)


    ###########################
    logger.info(f"\nDatabase complete!")
    logger.info(f"Downsweep calls: {n_downsweep} -> {ds_dir}")
    logger.info(f"Pulse calls: {n_pulse} ->{pulse_dir}")
    logger.info(f"Metadata CSV: {csv_path}")
    return csv_path