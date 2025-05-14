import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss, title='Loss Plot'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_waveforms(original_waveform, predicted_waveform, title='Waveform Comparison'):
    plt.figure(figsize=(10, 5))
    plt.plot(original_waveform, label='Original Waveform', color='blue')
    plt.plot(predicted_waveform, label='Predicted Waveform', color='orange')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def plot_spectrogram(spectrogram, title='Spectrogram'):
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()