import os
import pandas as pd
from functools import partial
from multiprocessing import Pool
import torchaudio
import warnings
from tqdm import tqdm
import torch
import tempfile
warnings.filterwarnings('ignore')

data_path = "data"

def get_audio_paths(metadata_path): 
    df = pd.read_csv(metadata_path)
    audio_paths =[os.path.join(df.iloc[i]['key'],df.iloc[i]['mp3name']) for i in range(len(df))]
    return audio_paths

failed_paths = []
def check_file(audio_path):
    try:
        audio_path = os.path.join(data_path,"raw_audio", audio_path)
        wav, _ = torchaudio.load(audio_path)
        aporee_id = str(audio_path.split('/')[-2])
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, aporee_id + '.pt')
            torch.save(wav, temp_file_path)        
            #print("Passed for:",audio_path)
    except:
        failed_paths.append(audio_path)
        print("Failed for:",audio_path)

audio_paths = get_audio_paths(os.path.join(data_path,"metadata.csv"))
audio_paths.sort()

for i in tqdm(range(len(audio_paths))):
    check_file(audio_paths[i])
    if i%1000 == 0:
        print("done for",i)
print(failed_paths)

#Save the id corresponding to corrupt mp3s 
ignore_ids = []
for f in failed_paths:
    fileid = str(f).split('/')[-2]
    ignore_ids.append(fileid)
df = pd.DataFrame(ignore_ids,columns=['key'])
df.to_csv(os.path.join(data_path,"corrupt_ids_final.csv"))


