import numpy as np
import librosa
import torch
import torchaudio

def compute_mel_spectrogram(audio_path, n_fft=1024, hop_length=256, n_mels=80):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    return mel_spectrogram

def kl_divergence(p, q):
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10  # Avoid log(0)
    return torch.sum(p * torch.log(p / q))

def l1_loss(original, synthesized):
    return torch.mean(torch.abs(original - synthesized))

def l2_loss(original, synthesized):
    return torch.mean((original - synthesized) ** 2)

def dynamic_time_warping(seq1, seq2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        seq1: First sequence (numpy array)
        seq2: Second sequence (numpy array)
        
    Returns:
        DTW distance
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = float('inf')
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    return dtw_matrix[n, m]