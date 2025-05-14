import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from src.data.dataset import PhonemeDataset
from src.model.tts_model import TTSModel
from src.config import Config

def train_model():
    # Load configuration
    config = Config()

    # Load metadata
    metadata = pd.read_csv(config.metadata_path, sep='|')
    
    # Create dataset and dataloader
    dataset = PhonemeDataset(metadata, config.audio_dir, config.phoneme_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    model = TTSModel(config)
    model.train()

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        for i, (phonemes, audio) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(phonemes)
            loss = criterion(outputs, audio)
            loss.backward()
            optimizer.step()

            if (i + 1) % config.log_interval == 0:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    train_model()