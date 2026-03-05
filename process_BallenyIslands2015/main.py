import os
import pandas as pd
import soundfile as sf
import numpy as np

from logging_config import setup_logging
from config import ABS_PATH_BallenyIslands2015, ABS_PATH_BP20HZ



def convert_txt_csv_file(logger=None):
    df = pd.read_csv(ABS_PATH_BP20HZ, sep='\t')
    output_csv = ABS_PATH_BP20HZ.replace('.txt', '.csv')
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved: {output_csv}")



def parse_selection_table(txt_file: str, logger=None) -> pd.DataFrame:
    df = pd.read_csv(txt_file, sep='\t')
    logger.info("df parsed")
    return df




def extract_pulse_clips(df: pd.DataFrame,wav_dir: str,output_dir: str,padding_s: float, logger=None) -> None:
    for _, row in df.iterrows():
        selection_id = int(row['Selection'])
        wav_filename = row['Begin File']
        begin_time_s = float(row['Begin Time (s)'])
        end_time_s = float(row['End Time (s)'])

        wav_path = os.path.join(wav_dir, wav_filename)
        if not os.path.isfile(wav_path):
            logger.warning(f"WAV not found, skipping: {wav_path}")
            continue
        else:
            logger.info(f"WAV file selected {wav_path}")
        

        with sf.SoundFile(wav_path) as wav_file:
            sr = wav_file.samplerate
            logger.info(f"Sample rate: {sr}")

            # convert times to samples, apply padding
            # start_sample = max(0, int((begin_time_s - padding_s) * sr))
            start_sample =int(row['Beg File Samp (samples)'])
            # end_sample = min(len(wav_file), int((end_time_s + padding_s) * sr))
            end_sample= int(row['End File Samp (samples)'])

            logger.info(f"Start sample at: {start_sample}")
            logger.info(f"End sample at: {end_sample}")


            # seek wav
            logger.info("Clipping the wav file!")
            wav_file.seek(start_sample)
            clip = wav_file.read(end_sample - start_sample)



        # saving clip
        logger.info("Saving the clip")
        out_name = f"sel{selection_id:04d}_{wav_filename}"
        out_path = os.path.join(output_dir, out_name)
        
        
        sf.write(out_path, clip, sr)
        logger.info(f"Saved: {out_path}  ({clip.shape[0]/sr:.2f}s)")




def main() -> None:
    logger = setup_logging()
    logger.info("")
    logger.info("Starting 20Hz pulse extraction")


    logger.info("")
    df = parse_selection_table(ABS_PATH_BP20HZ, logger)
    logger.info(f"Loaded {len(df)} selections")


    wav_dir = os.path.join(ABS_PATH_BallenyIslands2015, 'wav')
    output_dir = os.path.join(ABS_PATH_BallenyIslands2015, 'pulses_20hz')

    extract_pulse_clips(df, wav_dir, output_dir, padding_s=60, logger=logger)


    logger.info("Done")


if __name__ == "__main__":
    main()