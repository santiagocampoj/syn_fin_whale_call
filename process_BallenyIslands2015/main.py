import os
import pandas as pd
import soundfile as sf
from logging_config import setup_logging
from config import ABS_PATH_BallenyIslands2015, ABS_PATH_BP20HZ




def convert_txt_to_csv(logger=None):
    df = pd.read_csv(ABS_PATH_BP20HZ, sep='\t')
    output_csv = ABS_PATH_BP20HZ.replace('.txt', '.csv')
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV: {output_csv}")



def parse_selection_table(txt_file: str, logger=None) -> pd.DataFrame:
    df = pd.read_csv(txt_file, sep='\t')
    logger.info(f"df parsed with {len(df)} rows")
    return df




def extract_pulse_clips(df: pd.DataFrame,wav_dir: str,output_dir: str,padding_s: float,logger=None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        selection_id = int(row['Selection'])
        wav_filename = row['Begin File']

        wav_path = os.path.join(wav_dir, wav_filename)
        if not os.path.isfile(wav_path):
            logger.warning(f"WAV not found, skipping: {wav_path}")
            continue
        else:
            logger.info(f"WAV file selected: {wav_path}")



        with sf.SoundFile(wav_path) as wav_file:
            sr = wav_file.samplerate
            total_samples = len(wav_file)
            logger.info(f"Sample rate: {sr}, Total samples: {total_samples}")


            #padding
            padding_samples = int(padding_s * sr)
            start_sample = max(0, int(row['Beg File Samp (samples)']) - padding_samples)
            end_sample = min(total_samples, int(row['End File Samp (samples)']) + padding_samples)

            logger.info(f"Start sample: {start_sample}, End sample: {end_sample}")

            # clippling
            wav_file.seek(start_sample)
            clip = wav_file.read(end_sample - start_sample)


        #save
        out_name = f"sel{selection_id:04d}_{wav_filename}"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, clip, sr)
        logger.info(f"Saved: {out_path}  ({clip.shape[0]/sr:.2f}s)")




def main() -> None:
    logger = setup_logging()
    logger.info("")
    logger.info("Starting 20Hz pulse extraction")
    logger.info("")

    convert_txt_to_csv(logger)

    df = parse_selection_table(ABS_PATH_BP20HZ, logger)
    logger.info(f"Loaded {len(df)} selections")

    wav_dir = os.path.join(ABS_PATH_BallenyIslands2015, 'wav')
    output_dir = os.path.join(ABS_PATH_BallenyIslands2015, 'pulses_20hz')

    extract_pulse_clips(df, wav_dir, output_dir, padding_s=5.0, logger=logger)



    logger.info("")
    logger.info("Done")



if __name__ == "__main__":
    main()