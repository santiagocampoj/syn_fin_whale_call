import csv
import os
import soundfile as sf



def build_snr_database(mixed_wav: str,labels_csv: str,fs: int,out_dir: str,padding_s: float = 0.5,logger=None) -> str:
    # looading the full recording
    mixed, file_fs = sf.read(mixed_wav, dtype="float32", always_2d=False)
    if file_fs != fs:
        raise ValueError(f"Expected fs={fs} but mixed WAV has fs={file_fs}")

    # open labels
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        events = list(reader)
    if logger:
        logger.info(f"Loaded {len(events)} injection events from '{os.path.basename(labels_csv)}'")



    # structure
    ds_dir    = os.path.join(out_dir, "downsweep")
    pulse_dir = os.path.join(out_dir, "pulse")
    os.makedirs(ds_dir,    exist_ok=True)
    os.makedirs(pulse_dir, exist_ok=True)





    padding_samples = int(padding_s * fs)
    total_samples   = len(mixed)
    metadata_rows = []
    counters = {"downsweep": 0, "pulse": 0}
    for event in events:
        call_type = event["call_type"]
        offset_s= float(event["offset_s"])
        duration_s = float(event["duration_s"])
        snr_db = float(event["snr_db"])
        actual_snr = float(event["actual_snr_db"])

        # converting to samples
        start_sample = int(offset_s * fs)
        end_sample = start_sample + int(duration_s * fs)

        # padding (clamped to recording boundaries)
        clip_start = max(0, start_sample - padding_samples)
        clip_end = min(total_samples, end_sample + padding_samples)
        clip = mixed[clip_start:clip_end]

        # filename
        idx = counters[call_type]
        fname = f"{call_type}_{idx:04d}.wav"
        fpath = os.path.join(ds_dir if call_type == "downsweep" else pulse_dir, fname)

        sf.write(fpath, clip, fs)
        counters[call_type] += 1




        # Compute actual offset of call within the clip (accounting for padding)
        call_offset_in_clip_s = round((start_sample - clip_start) / fs, 4)
        metadata_rows.append({
            "filename":             os.path.join(call_type, fname),
            "call_type":            call_type,
            "clip_duration_s":      round((clip_end - clip_start) / fs, 4),
            "call_offset_in_clip_s": call_offset_in_clip_s,
            "call_duration_s":      round(duration_s, 4),
            "offset_in_ocean_s":    round(offset_s, 4),
            "snr_db":               round(snr_db, 2),
            "actual_snr_db":        round(actual_snr, 2),
            "padding_s":            padding_s,
            "fs":                   fs,
        })
        if logger:
            logger.info(f"Saved {call_type:10s} clip {idx:4d}  "
                        f"ocean_offset={offset_s:.2f}s  "
                        f"SNR={actual_snr:.1f} dB  -> {fname}")

    
    



    ###########################
    # save
    csv_out = os.path.join(out_dir, "metadata.csv")
    fieldnames = [
        "filename", "call_type", "clip_duration_s", "call_offset_in_clip_s",
        "call_duration_s", "offset_in_ocean_s", "snr_db", "actual_snr_db",
        "padding_s", "fs",
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    if logger:
        logger.info(f"SNR database complete!")
        logger.info(f"Downsweep clips: {counters['downsweep']}: {ds_dir}")
        logger.info(f"Pulse clips: {counters['pulse']}: {pulse_dir}")
        logger.info(f"Metadata CSV: {csv_out}")

    return csv_out