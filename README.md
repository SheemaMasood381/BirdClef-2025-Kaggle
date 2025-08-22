# 🐦 BirdCLEF 2025 - Soundscape Classification

This repository contains my experiments and solutions for the [BirdCLEF 2025 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2025). The goal is to identify bird species from long soundscape recordings using machine learning and audio signal processing techniques.

---

## Project Highlights
- **Custom PyTorch Model**: Based on `seresnext26t_32x4d` backbone using the TIMM library.  
- **Voice Activity Detection (VAD)**: Filters out non-bird audio segments to focus on relevant parts.  
- **Audio Preprocessing**: 
  - 10-second chunking of soundscape recordings for efficient inference.  
  - Log-mel spectrograms computed and stored as `.npz` batches.  
- **Self-Supervised Learning**: Pseudo-labeling applied using confident predictions from unlabeled soundscapes.  
- **Efficient Data Handling**: Custom `DataLoader` and `Dataset` with optional augmentations such as time/frequency masking and noise injection.  
- **Clean Label Management**: Generated pseudo-labels saved as `pseudo_labels_soundscape.csv` for downstream training.

---

## Dataset
The dataset consists of **long-form audio recordings of bird soundscapes**, provided by the BirdCLEF 2025 Kaggle Competition. Audio files are processed into 10-second chunks, and mel spectrogram features are extracted for model input.

---


## Installation

1. Clone the repository:

```bash
git clone https://github.com/SheemaMasood381/BirdCLEF-2025.git
cd BirdCLEF-2025
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Dependencies include:** PyTorch, torchaudio, timm, librosa, numpy, pandas, scikit-learn

---

## Usage

1. Preprocess the audio recordings (chunking & spectrograms).

2. Train the model:

```bash
python train.py --config configs/train_config.yaml
```

3. Inference on new soundscape data:

```bash
python inference.py --audio_path path/to/audio
```

4. Pseudo-labeling (optional for self-supervised learning):

```bash
python pseudo_labeling.py --unlabeled_data path/to/unlabeled
```

---

## Results

* Custom `seresnext26t_32x4d` model achieved strong performance on BirdCLEF validation data.
* Pseudo-labeling improved accuracy by leveraging unlabeled soundscape recordings.

*(Specific leaderboard scores can be included if permitted.)*

---

## Future Work

* Experiment with other transformer-based architectures for audio classification.
* Implement ensemble models combining multiple backbones for higher accuracy.
* Explore advanced data augmentation techniques to improve generalization.

---

## Acknowledgements

* BirdCLEF 2025 Kaggle Competition
* PyTorch & TIMM library
* Open-source audio processing tools: `librosa`, `torchaudio`

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

cd BirdCLEF-2025

