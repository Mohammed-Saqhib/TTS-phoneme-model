import torch
import torchaudio
from model.tts_model import TTSModel
from config import Config

def load_model(model_path):
    model = TTSModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def synthesize_audio(model, phoneme_input):
    with torch.no_grad():
        audio_output = model(phoneme_input)
    return audio_output

def save_audio(output, output_path):
    torchaudio.save(output_path, output, sample_rate=Config.SAMPLE_RATE)

def main(phoneme_input_path, model_path, output_audio_path):
    # Load the phoneme input
    with open(phoneme_input_path, 'r') as f:
        phoneme_input = f.read().strip()

    # Convert phoneme input to tensor
    phoneme_tensor = torch.tensor([phoneme_input])  # Adjust based on your preprocessing

    # Load the trained model
    model = load_model(model_path)

    # Synthesize audio from phoneme input
    audio_output = synthesize_audio(model, phoneme_tensor)

    # Save the synthesized audio
    save_audio(audio_output, output_audio_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phoneme-based TTS Inference')
    parser.add_argument('--phoneme_input', type=str, required=True, help='Path to the phoneme input file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_audio', type=str, required=True, help='Path to save the synthesized audio')

    args = parser.parse_args()
    main(args.phoneme_input, args.model_path, args.output_audio)