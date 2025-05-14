# Configuration settings for the TTS model
import torch

class Config:
    def __init__(self):
        # Model hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.hidden_size = 256
        self.num_layers = 2
        
        # File paths
        self.metadata_path = 'data/metadata.csv'
        self.audio_dir = 'data/audio/'
        self.phoneme_dir = 'data/preprocessed/phonemes/'
        self.preprocessed_phonemes_dir = 'data/preprocessed/phonemes/'
        self.preprocessed_spectrograms_dir = 'data/preprocessed/spectrograms/'
        
        # Model parameters
        self.input_dim = 128  # Phoneme embedding dimension
        self.hidden_dim = 256
        self.output_dim = 80  # Mel spectrogram dimension
        self.model_save_path = 'models/tts_model.pth'
        self.log_interval = 10
        
        # Evaluation settings
        self.evaluation_metrics = ['L1', 'L2', 'KL']
        
        # Device settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'