import os
import pandas as pd
import torchaudio
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Define the data path
raw_audio_path = "data/raw_audio"
metadata_path = "data/final_metadata.csv"
output_dir = "data/raw_audio_tensorized"

def get_audio_paths(metadata_path):
    """ Reads the CSV file and generates paths and short_keys for audio files. """
    df = pd.read_csv(metadata_path)
    # Creating a list of tuples (audio_path, short_key)
    audio_info = [(os.path.join(raw_audio_path, df.iloc[i]['key'], df.iloc[i]['mp3name']), df.iloc[i]['short_key']) for i in range(len(df))]
    return audio_info

def process_audio_file(audio_info):
    """ Processes each audio file, converting it to a tensor and saving with short_key as filename. 
        Throws an error and crashes if the file cannot be processed.
    """
    audio_path, short_key = audio_info
    wav, _ = torchaudio.load(audio_path)  # This will throw an error if the file cannot be processed
    temp_file_path = os.path.join(output_dir, f"{short_key}.pt")
    torch.save(wav, temp_file_path)

def main():
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Get paths and short_keys to audio files from metadata
    audio_info = get_audio_paths(metadata_path)
    audio_info.sort(key=lambda x: x[1])  # Sort by short_key if needed

    # Processing audio files
    for info in tqdm(audio_info):
        process_audio_file(info)  # Any error here will crash the program

    # Success message
    print("\nSuccess! All samples have been successfully converted to tensors and saved.")
    print("The processed files are located in the directory: '{}'".format(output_dir))

if __name__ == "__main__":
    main()
