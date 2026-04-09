# Unsupervised Neural Network for Multi-Genre Music Generation

## Overview
This project implements unsupervised neural network methods for symbolic music generation using MIDI data. The work is based on the course project **“Unsupervised Neural Network for Multi-Genre Music Generation”**. The pipeline covers preprocessing, sequence modeling, generation, and evaluation using piano-roll representations. :contentReference[oaicite:1]{index=1}

## Dataset
We used the **MAESTRO v3.0.0 MIDI** dataset for initial experiments and pipeline development.
- Metadata CSV used for train/validation/test split
- MIDI files converted to piano-roll sequences
- Pitch range cropped to piano keys (88 notes)
- Fixed-length sequence windows created for training

## Project Structure
```text
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/
│   ├── processed/
│   └── train_test_split/
├── notebooks/
├── src/
│   ├── config.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── generation/
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
├── report/
└── architecture_diagrams/

```
Download Official dataset: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

Implemented Components
1. Preprocessing
Metadata loading and split generation
MIDI parsing with pretty_midi
Piano-roll conversion
Fixed-length sequence segmentation
2. Task 1: LSTM Autoencoder

Implemented an LSTM-based autoencoder for piano-roll reconstruction.

Training on preprocessed MAESTRO sequences
MIDI reconstruction export
Quantitative evaluation using pitch histogram and rhythm metrics
3. Task 2: Variational Autoencoder (VAE)

Implemented a sequence VAE with:

LSTM encoder
latent mean and log-variance
reparameterization trick
LSTM decoder
latent sampling for new MIDI generation
Evaluation Metrics

The following metrics are used:

Pitch Histogram Distance
Rhythm Diversity Score
Repetition Ratio
Key Results
Autoencoder

The AE baseline successfully reconstructed MIDI-like outputs, but showed limited input-specific diversity and tended toward repetitive rhythm patterns.

VAE

The VAE successfully generated 8 sample MIDI outputs from latent sampling. However, the generated samples remained conservative and showed limited diversity, suggesting partial posterior-collapse-like behavior.

How to Run
1. Preprocessing
PYTHONPATH=. ./.venv/bin/python -m src.preprocessing.midi_parser
PYTHONPATH=. ./.venv/bin/python -m src.preprocessing.piano_roll
2. Train Autoencoder
PYTHONPATH=. ./.venv/bin/python -m src.training.train_ae
3. Generate AE Reconstructions
PYTHONPATH=. ./.venv/bin/python -m src.generation.generate_music
4. Train VAE
PYTHONPATH=. ./.venv/bin/python -m src.training.train_vae
5. Sample from VAE
PYTHONPATH=. ./.venv/bin/python -m src.generation.sample_latent
6. Run Evaluation
PYTHONPATH=. ./.venv/bin/python -m src.evaluation.metrics
Outputs

Generated files are stored in:

outputs/generated_midis/ae_reconstructions/
outputs/generated_midis/vae_generated/
outputs/plots/
Limitations
Experiments were performed on MAESTRO, which is mostly classical piano rather than truly multi-genre.
AE and VAE outputs showed limited diversity.
More expressive decoding and stronger latent regularization may improve generation quality.
Future Work
Add a true multi-genre dataset such as Lakh MIDI
Improve VAE latent utilization
Add Transformer-based generation
Add human listening evaluation
