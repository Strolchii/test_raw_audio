# This script attempts to perform post-processing of 'description' column 
# of metadata of the SoundingEarth dataset.
import re
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.point import Point
from cleantext import clean     #pip install clean-text (cleantext is a different package)
import os
# from wordsegment import load, segment
# import nltk
from tqdm import tqdm
# nltk.download('words')
# words = set(nltk.corpus.words.words())
import sys
from config import cfg

geolocator = Nominatim(user_agent="openmapquest")
def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse(Point(lat, lon),language='en')
        address = location.address
        return address
    except:
        address = None
        return address

# def splitwords(word):
#     load()
#     return ' '.join(segment(word))

def clean_description(description):
    sent = re.sub(r'(<br\s*/>)',' ',description)
    output = clean(sent,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks= True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                   # replace all URLs with a special token
        no_emails=True,                 # replace all email addresses with a special token
        no_phone_numbers=True,          # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct= True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
                )

    output = re.sub(r'\s+',' ',output)

    return output


def get_caption(i, lat, lon, title, description):
    if pd.notna(description):
        description = description
    else:
        description = title
    
    address = reverse_geocoding(lat=lat, lon=lon)
    if address != None:
        caption = clean_description(description + '. The location of the sound is: '+address+'.')
    else:
        caption = clean_description(description + '.')
    if i%100 == 0:
        print(str(i)+" done!")
    return caption


def get_detailed_metadata(data_path):
    meta_df = pd.read_csv(os.path.join(cfg.data_path,'metadata.csv'))
    print("Original data",len(meta_df))
    corrupt_ids = list(pd.read_csv(os.path.join(cfg.data_path,"corrupt_ids_final.csv"))['key']) #Ignore some IDs whose mp3 is found to be corrupt
    meta_df = meta_df[~meta_df['key'].isin(corrupt_ids)]
    print("count of mp3 after removing corrupt mp3",len(meta_df))

    #New: Saving ids of audio files < 16000
    meta_df_low_sr = meta_df[meta_df['mp3samplerate'] < 16000]
    print("data removing mp3 sampled by sr less than 16k", len(meta_df_low_sr))

    audio_low_sr_ids = meta_df_low_sr['key'].tolist()
    low_sr_ids_df = pd.DataFrame(audio_low_sr_ids, columns=['key'])
    low_sr_ids_df.to_csv(os.path.join(cfg.data_path, 'sr_less_16k_ids_final.csv'))

    #keep only the data with audio fs >= 16000
    meta_df = meta_df[meta_df['mp3samplerate']>=16000]
    print("count of mp3 after removing mp3 sampled by sr less than 16k",len(meta_df))
    audio_short_ids = list(meta_df.key)

    image_ids = os.listdir(cfg.sat_image_path)
    image_short_ids = [i.split('.jpg')[0] for i in image_ids]
    print("TOTAL COUNT of sat image samples in original dataset", len(image_short_ids)) 

    metadata = meta_df.fillna(np.nan)
    
    # only for debug use
    # metadata = metadata.head(200)

    keys = list(metadata.key)
    lats = list(metadata.latitude)
    longs = list(metadata.longitude)
    titles = list(metadata.title)
    descriptions = list(metadata.description)

    captions = [get_caption(i, lats[i], longs[i], titles[i], descriptions[i]) for i in range(len(metadata))]
    metadata['caption'] = captions

    # Identifying entries with missing address details
    no_address_ids = []
    address = []
    for i, caption in enumerate(captions):
        if "location of the sound is" in caption:
            address.append("The location of the sound is" + caption.split("location of the sound is")[1])
        else:
            no_address_ids.append(keys[i])
            address.append("") # Add empty adress field
    metadata['address'] = address
    metadata.reset_index(drop=True, inplace=True)

    no_address_ids_df = pd.DataFrame(no_address_ids, columns=['key'])
    no_address_ids_df.to_csv(os.path.join(cfg.data_path, 'no_address_ids_final.csv'))
    print("count of mp3 with no reversed geocode",len(no_address_ids_df)) # dont remove these !!!!

    # Save final metadata before adding captions, excluding samples without addresses
    # valid_metadata = metadata[~metadata['key'].isin(no_address_ids)]   # <- deactivated to use all samples, that means dont remove the samples without address information
    valid_metadata = metadata
    valid_metadata.reset_index(drop=True, inplace=True)

    valid_metadata.to_csv(os.path.join(cfg.data_path, 'final_metadata.csv'))
    print("TOTAL COUNT of valid sounds in dataset", len(valid_metadata)) 

    print("Description of metadata cleaned")
    return valid_metadata

get_detailed_metadata(cfg.data_path)
