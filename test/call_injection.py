"""
  1. Load a chunk (e.g. 1 min) from the real recording
  2. Resample the background to match the synthetic signal fs if needed
  3. For each synthetic call:
       - Pick a random time offset within the recording
       - Measure local background RMS in that window
       - Scale the call to achieve the desired SNR
       - Add it in place
  4. Save the mixed recording + a metadata CSV with injection timestamps and SNR
"""

import numpy as np
import soundfile as sf
import csv
import os
from typing import Optional

from scipy.signal import resample_poly
from math import gcd

from class_params import InjectionEvent




def _rms(x: np.ndarray) -> float:
    """Root mean square of a signal."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _resample(audio: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """
    Simple resample using scipy.  Only called when fs mismatch is detected.
    """
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(audio, up, down).astype(np.float32)







def load_background(wav_path: str,fs_target: int,chunk_start_s: float = 0.0,chunk_dur_s: float = 60.0,logger=None) -> tuple[np.ndarray, int]:
    info = sf.info(wav_path)
    orig_fs = info.samplerate
    total_dur = info.duration

    # clamp chunk to available audio
    chunk_start_s = max(0.0, min(chunk_start_s, total_dur))
    chunk_dur_s = min(chunk_dur_s, total_dur - chunk_start_s)

    start_frame = int(chunk_start_s * orig_fs)
    n_frames = int(chunk_dur_s * orig_fs)

    audio, _ = sf.read(wav_path, start=start_frame, frames=n_frames,
                       dtype="float32", always_2d=False)



    #  mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if logger:
        logger.info(f"Loaded {chunk_dur_s:.1f}s from '{os.path.basename(wav_path)}' "
                    f"(orig fs={orig_fs} Hz, frames={len(audio)})")



    # resample if necessary
    if orig_fs != fs_target:
        if logger:
            logger.info(f"Resampling {orig_fs} Hz -> {fs_target} Hz ...")
        audio = _resample(audio, orig_fs, fs_target)
        if logger:
            logger.info(f"Resampled length: {len(audio)} samples")

    return audio, fs_target






def inject_call(background: np.ndarray,call: np.ndarray,offset_samples: int,snr_db: float) -> tuple[np.ndarray, float, float]:
    """
    Insert a single call into the background at a given offset and SNR.

    The call is scaled so that:
        SNR (dB) = 20 * log10(call_rms / background_rms_in_window)
    """
    mixed = background.copy()
    call_len = len(call)
    end = offset_samples + call_len

    # Safety: clip if call extends past end of recording
    if end > len(mixed):
        call = call[:len(mixed) - offset_samples]
        call_len = len(call)
        end = offset_samples + call_len

    # Local background RMS in the injection window
    bg_window = mixed[offset_samples:end]
    bg_rms = _rms(bg_window)

    # Scale call to achieve target SNR
    call_rms = _rms(call)
    target_call_rms = bg_rms * (10 ** (snr_db / 20.0))
    scale_factor = target_call_rms / (call_rms + 1e-12)
    scaled_call = call * scale_factor

    mixed[offset_samples:end] += scaled_call

    # Measure actual SNR
    actual_snr_db = 20 * np.log10(_rms(scaled_call) / (bg_rms + 1e-12))

    return mixed, float(scale_factor), float(actual_snr_db)






# main injection
def inject_calls_into_recording(
    background: np.ndarray,
    fs: int,
    calls: list[tuple[str, np.ndarray]],   # list ofcall_type, audio_array
    snr_db_range: tuple[float, float] = (5.0, 20.0),
    min_gap_s: float = 2.0,
    seed: Optional[int] = None,
    logger=None,
) -> tuple[np.ndarray, list[InjectionEvent]]:
    """
    Inject multiple synthetic calls into a background recording.

    Args:
        background:    1-D float32 array of the background recording.
        fs:            Sample rate (Hz).
        calls:         List of (call_type_str, audio_array) tuples to inject.
        snr_db_range:  (min_snr, max_snr) — each call gets a random SNR in this range.
        min_gap_s:     Minimum silence gap between injected calls (seconds).
        seed:          Random seed for reproducibility.
        logger:        Optional logger.

    Returns:
        mixed:   The background with all calls injected.
        events:  List of InjectionEvent describing each injection.
    """


    if seed is not None:
        np.random.seed(seed)

    bg_dur_s = len(background) / fs
    mixed = background.copy()
    events: list[InjectionEvent] = []

    # track occupied regions to avoid overlapping calls
    occupied: list[tuple[int, int]] = []

    min_gap_samples = int(min_gap_s * fs)

    for call_type, call in calls:
        call_len = len(call)
        call_dur_s = call_len / fs

        # try to find a non-overlapping offset (max 200 attempts)
        placed = False
        for _ in range(200):
            max_start = len(background) - call_len - 1
            if max_start <= 0:
                break
            offset = np.random.randint(0, max_start)
            end = offset + call_len

            # check against already-placed calls + required gap
            overlap = any(
                not (end + min_gap_samples <= s or offset >= e + min_gap_samples)
                for s, e in occupied
            )
            if not overlap:
                placed = True
                break

        if not placed:
            if logger:
                logger.warning(f"Could not place {call_type} call without overlap — skipping.")
            continue



        # random SNR within specified range
        snr_db = np.random.uniform(*snr_db_range)

        mixed, scale, actual_snr = inject_call(mixed, call, offset, snr_db)
        occupied.append((offset, end))

        event = InjectionEvent(
            call_type=call_type,
            offset_s=round(offset / fs, 4),
            duration_s=round(call_dur_s, 4),
            snr_db=round(snr_db, 2),
            actual_snr_db=round(actual_snr, 2),
            scale_factor=round(scale, 6),
        )
        events.append(event)

        if logger:
            logger.info(f"  Injected {call_type:10s} at {event.offset_s:6.2f}s  "
                        f"SNR={event.actual_snr_db:.1f} dB  scale={event.scale_factor:.4f}")

    # sort events by time
    events.sort(key=lambda e: e.offset_s)
    return mixed, events









def save_injection_results(mixed: np.ndarray,fs: int,events: list[InjectionEvent],out_dir: str,out_name: str = "mixed_recording",logger=None) -> tuple[str, str]:
    """
    Save the mixed WAV file and a metadata CSV of injection events.
    """
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, f"{out_name}.wav")
    csv_path = os.path.join(out_dir, f"{out_name}_labels.csv")

    sf.write(wav_path, mixed, fs)

    fieldnames = ["call_type", "offset_s", "duration_s", "snr_db",
                  "actual_snr_db", "scale_factor"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            writer.writerow({
                "call_type":    e.call_type,
                "offset_s":     e.offset_s,
                "duration_s":   e.duration_s,
                "snr_db":       e.snr_db,
                "actual_snr_db": e.actual_snr_db,
                "scale_factor": e.scale_factor,
            })

    if logger:
        logger.info(f"Saved mixed WAV  : {wav_path}")
        logger.info(f"Saved labels CSV : {csv_path}")

    return wav_path, csv_path