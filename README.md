This repository contains my experiments and solutions for the [BirdCLEF 2025 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2025). The goal is to identify bird species from long soundscape recordings using machine learning and audio signal processing techniques.

---

## 🧠 Project Highlights

- ✅ **Custom PyTorch Model** based on `seresnext26t_32x4d` backbone using [TIMM](https://huggingface.co/timm).
- ✅ **VAD Filtering**: Voice Activity Detection applied to filter non-bird segments.
- ✅ **10-second Chunking**: Soundscape audio split into 10s chunks for inference.
- ✅ **Mel Spectrograms**: Log-mel spectrograms computed and saved as `.npz` batches.
- ✅ **Pseudo-Labeling**: Self-supervised learning using confident predictions on unlabeled soundscapes.
- ✅ **DataLoader + Dataset**: Efficient loading from `.npz` with optional augmentations like time/freq masking and noise.
- ✅ **Clean Label Merging**: Generated pseudo-labels saved to `pseudo_labels_soundscape.csv` for later training.

---
