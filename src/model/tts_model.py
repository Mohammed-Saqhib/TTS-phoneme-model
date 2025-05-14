from torch import nn
import torch

class TTSModel(nn.Module):
    def __init__(self, config=None):
        super(TTSModel, self).__init__()
        if config is None:
            # Default parameters if config is not provided
            input_dim = 128
            hidden_dim = 256
            output_dim = 80
        else:
            # Use config parameters
            input_dim = config.input_dim
            hidden_dim = config.hidden_dim
            output_dim = config.output_dim
            
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def generate_audio(self, phoneme_sequence):
        with torch.no_grad():
            output = self.forward(phoneme_sequence)
            return output