import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf

def load_metadata(metadata_path):
    """Load metadata from a pipe-separated CSV file."""
    metadata = pd.read_csv(metadata_path, sep='|')
    return metadata

def preprocess_audio(audio_path, target_sr=22050):
    """Load and preprocess audio file."""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr

def extract_phonemes(text):
    """Dummy phoneme extraction function."""
    # This should be replaced with a proper phoneme extraction method
    return list(text)

def save_preprocessed_data(phonemes, audio, output_dir, filename):
    """Save preprocessed phoneme and audio data."""
    phoneme_path = os.path.join(output_dir, 'phonemes', f'{filename}.npy')
    audio_path = os.path.join(output_dir, 'spectrograms', f'{filename}.wav')
    
    np.save(phoneme_path, phonemes)
    sf.write(audio_path, audio, 22050)

def preprocess_data(metadata_path, audio_dir, output_dir):
    """Main function to preprocess data."""
    metadata = load_metadata(metadata_path)
    
    for index, row in metadata.iterrows():
        text = row['text']
        audio_file = row['audio_file']
        audio_path = os.path.join(audio_dir, audio_file)
        
        audio, sr = preprocess_audio(audio_path)
        phonemes = extract_phonemes(text)
        
        save_preprocessed_data(phonemes, audio, output_dir, os.path.splitext(audio_file)[0])

# Example usage
# preprocess_data('data/metadata.csv', 'data/audio', 'data/preprocessed')