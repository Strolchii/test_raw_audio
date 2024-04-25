import pandas as pd
import os

# Pfade zu den CSV-Dateien
corrupt_ids_path = 'data/SoundingEarth/data/corrupt_ids_final.csv'
metadata_path = 'data/SoundingEarth/data/metadata.csv'

# CSVs einlesen
corrupt_ids = pd.read_csv(corrupt_ids_path)
metadata = pd.read_csv(metadata_path)

# Nur die Einträge aus 'metadata' behalten, die in 'corrupt_ids' vorhanden sind
corrupted_metadata = metadata[metadata['key'].isin(corrupt_ids['key'])]

# Verzeichnis für die Downloads erstellen
os.makedirs('data/SoundingEarth/data/raw_audio_cor_reload', exist_ok=True)

# In das Verzeichnis wechseln
os.chdir('data/SoundingEarth/data/raw_audio_cor_reload')

# Download-Befehl für jedes korrupte Audio vorbereiten
for key in corrupted_metadata['key']:
    # Hier wird das Internet Archive Utility (ia) zum Download verwendet
    # Dieses Skript setzt voraus, dass `ia` installiert und konfiguriert ist
    os.system(f'/usr/bin/bash -c "ia download {key} --glob=\'*.mp3\'"')
