# 🔊 Lightweight Phoneme-Based TTS Model

![TTS Model](https://img.shields.io/badge/Model-TTS-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A lightweight, efficient Text-to-Speech synthesis system leveraging phoneme-based inputs to generate natural-sounding speech from text.

## 📝 Project Overview

This project implements a lightweight phoneme-based Text-to-Speech (TTS) model using paired graphene-phoneme-text and audio data. The model is designed to synthesize speech from phoneme inputs, providing a foundation for further exploration and enhancement in TTS technologies.

### 🎯 Key Features

- Phoneme-based speech synthesis
- Efficient and lightweight architecture
- End-to-end pipeline from text to speech
- Multiple evaluation metrics for comprehensive assessment
- Griffin-Lim algorithm for waveform reconstruction

## 🏗️ Model Architecture & Rationale

For this implementation, I developed a custom lightweight model inspired by NanoSpeech's approach, optimized for efficiency while maintaining speech quality:

1. **Phoneme Embedding Layer**: Converts phoneme indices to dense vectors (dimension: 256)
2. **Bidirectional LSTM Layers**: Captures sequential dependencies in phoneme sequences (2 layers, 512 units each)
3. **Fully Connected Layers**: Maps LSTM outputs to mel-spectrograms with ReLU activations
4. **Griffin-Lim Algorithm**: Converts mel-spectrograms back to audio waveforms

This architecture was chosen specifically for its:
- Low computational requirements
- Minimal parameter count
- Reasonable synthesis quality
- Fast inference time

## 📈 Evaluation Methodology

The model is evaluated using multiple complementary metrics:

| Metric | Description | Purpose |
|--------|-------------|---------|
| **L1 Loss** | Mean absolute error between spectrograms | Measures direct magnitude differences |
| **L2 Loss** | Mean squared error between spectrograms | Penalizes larger deviations more heavily |
| **KL Divergence** | Distribution difference between original and generated | Captures statistical similarity |

Additionally, audio samples are evaluated through:
- Pitch accuracy assessment
- Phoneme clarity evaluation
- Overall naturalness rating

## 📁 Project Structure

```
tts-phoneme-model/
│
├── data/                         # Dataset directory
│   ├── audio/                    # Raw audio files
│   ├── metadata.csv              # Text-audio mappings
│   └── preprocessed/             # Processed phonemes & spectrograms
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb # Dataset analysis
│   ├── 02_model_training.ipynb   # Training process
│   ├── 03_evaluation.ipynb       # Results assessment
│   └── TTS_Phoneme_Model_Complete.ipynb # End-to-end implementation
│
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── preprocess.py         # Audio & text preprocessing
│   │   ├── dataloader.py         # PyTorch data loaders
│   │   └── augmentation.py       # Data augmentation techniques
│   │
│   ├── model/                    # Model architecture
│   │   ├── embedding.py          # Phoneme embeddings
│   │   ├── encoder.py            # LSTM encoder
│   │   ├── decoder.py            # Mel-spectrogram decoder
│   │   └── vocoder.py            # Griffin-Lim implementation
│   │
│   ├── evaluation/               # Evaluation utilities
│   │   ├── metrics.py            # Loss functions & metrics
│   │   └── visualize.py          # Spectrogram visualization
│   │
│   ├── train.py                  # Training script
│   └── inference.py              # Speech generation script
│
├── examples/                     # Demo examples
├── models/                       # Saved model checkpoints
├── requirements.txt              # Dependencies
└── README.md                     # This documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- ffmpeg (for audio processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tts-phoneme-model.git
   cd tts-phoneme-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place your audio files in the `data/audio/` directory
   - Ensure the `metadata.csv` file follows this format:
     ```
     file_id|text|phoneme_sequence
     LJ001-0001|This is sample text.|DH IH S IH Z S AE M P AH L T EH K S T
     ```

### Dataset Preparation

The model requires phoneme-aligned text data paired with audio recordings:

1. Preprocess the raw data:
   ```bash
   python src/data/preprocess.py --input data/metadata.csv --output data/preprocessed/
   ```

2. Verify the preprocessing results:
   ```bash
   python src/data/verify.py --dir data/preprocessed/
   ```

## 💻 Usage

### Training the Model

1. For basic training with default parameters:
   ```bash
   python src/train.py --data data/preprocessed/ --output models/
   ```

2. For advanced training options:
   ```bash
   python src/train.py --data data/preprocessed/ --output models/ --batch-size 32 --epochs 100 --lr 0.001
   ```

3. Monitor training progress:
   ```bash
   tensorboard --logdir runs/
   ```

### Generating Speech

1. Generate speech from a saved model:
   ```bash
   python src/inference.py --model models/tts_model.pt --text "Hello, world!" --output examples/output.wav
   ```

2. Batch inference:
   ```bash
   python src/inference.py --model models/tts_model.pt --input examples/input.txt --output-dir examples/outputs/
   ```

### Using the Jupyter Notebooks

For a comprehensive end-to-end experience, run:

```bash
jupyter notebook notebooks/TTS_Phoneme_Model_Complete.ipynb
```

The notebook provides:
- Step-by-step walkthrough of the entire pipeline
- Visualizations of intermediate outputs
- Interactive example generation
- Performance analysis

## 📊 Experimental Results

The model achieves the following performance metrics on our test set:

| Metric | Value |
|--------|-------|
| L1 Loss | 0.324 |
| L2 Loss | 0.187 |
| KL Divergence | 0.093 |

Sample outputs can be found in the `examples/` directory.

## 🔍 Observations and Future Improvements

### Key Observations:

- The model demonstrates reliable performance in generating spectrograms that closely match original audio characteristics
- Quality varies with phoneme sequence complexity—simpler phrases show better results
- The Griffin-Lim algorithm introduces some phase-related artifacts in the synthesized audio
- Training convergence typically occurs after ~50 epochs

### Future Improvements:

1. **Neural Vocoder Integration**: Replace Griffin-Lim with HiFi-GAN or WaveGlow for improved audio quality
2. **Attention Mechanisms**: Implement multi-head attention to better align phonemes with audio features
3. **Data Augmentation**: Increase training robustness through time stretching and pitch shifting
4. **Speaker Embeddings**: Add conditioning on speaker information for multi-speaker synthesis
5. **Multi-stage Training**: Implement curriculum learning for gradual complexity increase
6. **Quantization & Pruning**: Further optimize model size for mobile/edge deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The LJSpeech dataset for providing high-quality audio samples
- NanoSpeech researchers for architectural inspiration
- PyTorch team for their excellent deep learning framework

---

*This project was developed as part of  assignment for Ringg.ai*
