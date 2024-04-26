import os
import librosa
import numpy as np
import pandas as pd
import concurrent.futures
from config import cfg

########## Define parameters ######################
mel_bins = 128
# 16 kHz for speech, 22.05 kHz as a compromise, 44.1 kHz for ambient sounds, None to keep the original sampling rate
sr_kHz = 22.05 
sr = sr_kHz * 1e3

output_folder = os.path.join(cfg.sat_audio_spectrograms_path, f"{mel_bins}mel_{sr_kHz}kHz")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read metadata
metadata_path = os.path.join(cfg.data_path, 'final_metadata_with_captions.csv')
df = pd.read_csv(metadata_path)

# Initialize calculation of total size of Mel spectrograms and total processed size
total_mel_size = 0
processed_files_size = 0
processed_files_count = 0
total_files_size = df['mp3mb'].sum() / 1024  # Convert from MB to GB

# Iterate through each row of metadata
for idx, row in df.iterrows():
    mp3_path = os.path.join(cfg.sat_audio_path, row['key'], row['mp3name'])
    output_path = os.path.join(output_folder, f"{row['short_key']}.npy")
    
    # Load MP3 and calculate Mel spectrogram
    audio, sr_audio = librosa.load(mp3_path, sr=sr)  # sr=None to keep the original sampling rate
    S = librosa.feature.melspectrogram(audio, sr=sr_audio, n_mels=mel_bins)
    
    # Save the Mel spectrogram as a NumPy array
    np.save(output_path, S)
    
    # Update statistics
    mel_size = os.path.getsize(output_path) / (1024 ** 3)  # Size in GB
    file_size_gb = os.path.getsize(mp3_path) / (1024 ** 3)  # Size in GB
    total_mel_size += mel_size
    processed_files_size += file_size_gb
    processed_files_count += 1
    
    # Print status every 5GB of processed audio data
    if processed_files_size >= 5:
        print(f"Processed {processed_files_size:.2f} GB of {total_files_size:.2f} GB audio files ({processed_files_count} files)")
        print(f"Total data size of saved arrays so far: {total_mel_size:.2f} GB")
        processed_files_size = 0  # Reset after each output

# Final output
print("Processing completed.")
print(f"Total data size of saved arrays: {total_mel_size:.2f} GB")
