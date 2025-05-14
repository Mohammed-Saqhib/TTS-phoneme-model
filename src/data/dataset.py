import pandas as pd
import os
from torch.utils.data import Dataset
import torchaudio

class PhonemeDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, transform=None):
        self.metadata = pd.read_csv(metadata_file, sep='|')
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        graphene_text = self.metadata.iloc[idx, 0]
        audio_file = self.metadata.iloc[idx, 1]
        audio_path = os.path.join(self.audio_dir, audio_file)

        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return {
            'graphene_text': graphene_text,
            'waveform': waveform,
            'sample_rate': sample_rate
        }