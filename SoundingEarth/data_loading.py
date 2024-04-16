import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

LOW  = np.exp(-15 / 10)
HIGH = np.exp(5 / 10)


class AporeeDataset(Dataset):
    def __init__(self, root='data/'):
        super().__init__()
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / 'metadata.csv')
        # self.meta.insert(1, 'short_key', self.meta.key.apply(lambda x: int(x.split('_')[-1])))

        # # join and merge
        # img_present = set(int(f.stem) for f in self.root.glob('images/*.jpg'))
        # snd_present = set(int(f.stem) for f in self.root.glob('spectrograms/*.jpg'))

        # for img in self.root.glob('images/*.jpg'):
        #     key = int(img.stem)
        #     if key not in snd_present:
        #         img.rename(f'data/images_bak/{img.stem}.jpg')

        # self.meta = self.meta[self.meta.short_key.isin(img_present) &
        #                       self.meta.short_key.isin(snd_present)]
        # self.meta = self.meta.reset_index(drop=True)
        # self.meta.to_csv('./meta_new.csv', index=False)
        # print('Number of Samples:', len(self.meta))

    def __getitem__(self, idx):
        sample = self.meta.iloc[idx]
        key = sample['short_key']

        img = np.array(Image.open(self.root / 'images' / f'{key}.jpg'))
        img = torch.from_numpy(img).permute(2, 0, 1)

        audio = np.array(Image.open(self.root / 'spectrograms' / f'{key}.jpg')).astype(np.float32)
        audio = audio * ((HIGH - LOW) / 255) + LOW

        # Cut random 128x128 patch from spectrogram
        if audio.shape[1] > 128:
            start = int(torch.randint(0, audio.shape[1] - 128, []))
            audio = audio[:, start:start+128]
        audio = audio[np.newaxis]

        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        coords = torch.from_numpy(np.stack([lat, lon])).float()

        return [key, img, audio, coords]

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    ds = AporeeDataset('./data/')
    loader = DataLoader(ds, batch_size=16, num_workers=8)
    from tqdm import tqdm
    for i, (key, img, audio, coords) in enumerate(tqdm(loader)):
        if i == 0:
            print('img', img.shape)
            print('audio', audio.shape)

