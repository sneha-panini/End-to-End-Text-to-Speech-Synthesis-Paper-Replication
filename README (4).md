# End-to-End Text to Speech Synthesis: Paper Replication

A replication of **"End-to-End Text to Speech Synthesis"** (Xiao Wang, Yahan Yang, Ye Li; Stanford CS 229, Autumn 2018). Three models (SVR, a simple neural network, and a Seq2Seq model with attention) learn to map text directly to a speech spectrogram, which is then converted to a `.wav` file.

**Authors**

| Name | SRN |
|---|---|
| Sujay Nayak | PES2UG24CS533 |
| Sneha Panini | PES2UG24CS508 |

---

## Overview

Traditional TTS systems chain several stages (text normalisation, encoder, decoder, vocoder). The end-to-end approach replaces them with one learned model:

```
Text  ->  CMU phoneme/letter mapping  ->  Model  ->  Spectrogram  ->  Inverse STFT + inverse FIR  ->  .wav
```

The goal of this project is to see how far simple models and a small Seq2Seq network get on this task, following the paper as closely as the paper and poster allow. Nothing was added that the paper does not describe (no log scaling, normalisation, PCA, Griffin-Lim, decoder noise, or length predictor).

## Repository contents

| File | Description |
|---|---|
| `text_to_speech_paper_replication.ipynb` | Full pipeline: preprocessing, the three models, postprocessing, plots, and MOS evaluation |
| `tts_replication_presentation_final.pptx` | Project presentation (12 slides) |

## Dataset

[LJ Speech](https://keithito.com/LJ-Speech-Dataset/): about 13,100 public-domain audiobook clips from a single speaker.

| Setting | Value |
|---|---|
| Training sentences | 200 (random, fixed seed) |
| Dev sentences | 20 (used only to report dev loss) |
| Clip length cap | 5 s (**our change**; the paper has no cap) |
| Sample rate | 22,050 Hz |

## Pipeline

### Text preprocessing
- Each word is looked up in the CMU dictionary and randomly mapped to **phonemes or letters** (probability 0.5 for phonemes), e.g. `bit` becomes `@B @IH @T` or `b i t`.
- Symbols are converted to integer IDs (vocabulary size 73) and zero-padded to the longest sentence.
- Result: `X` of shape **220 x 88** (sentences x symbols).

### Audio preprocessing
- FIR low-pass filter (101 taps, 8 kHz cutoff), then STFT (2048-point window, 1025 frequency bins), keeping **magnitude only**.
- Zero-padded to 109 frames. Raw amplitudes, no log scaling (max 0.199, mean 0.00026).
- Result: `Y` of shape **220 x 109 x 1025**.

### Postprocessing
Predicted magnitude -> inverse STFT (zero phase) -> inverse FIR filter (regularised frequency-domain inverse) -> normalise -> save as 16-bit `.wav`.

## Models

| Model | Description |
|---|---|
| **SVR** | One polynomial-kernel SVR per time step and frequency bin (scikit-learn defaults). CPU only and very slow. |
| **Simple NN** | Embedding -> Flatten -> Dense (sigmoid) -> Dense (sigmoid, `T x 1025`) -> Reshape. About 39.6 M parameters. Batch size 10. |
| **Seq2Seq + attention** | Bidirectional LSTM encoder (512 latent dims), LSTM decoder with attention, trained with the target shifted one frame ahead. At generation time the decoder is seeded with a first frame and produces the rest frame by frame. |

Training: Adam optimiser, MSE loss, 100 epochs (2,000 iterations with batch size 10).

## Results

Naturalness was rated by 5 listeners on the Mean Opinion Score (MOS) scale, 1 (bad) to 5 (excellent).

| Model | Paper MOS | Our MOS | Std | Listeners |
|---|---|---|---|---|
| SVR | 1.0 | 1.0 | 0.00 | 5 |
| Simple NN | 1.7 | 1.4 | 0.49 | 5 |
| Seq2Seq | 2.5 | 1.6 | 0.49 | 5 |

The ordering matches the paper (Seq2Seq > NN > SVR), but the absolute scores are lower and, with only 5 listeners, the gaps are small.

### Why each model fell short
No model beats simply predicting all zeros in terms of loss.

- **SVR:** near-constant output, because the default epsilon is larger than the spectrogram amplitudes.
- **Simple NN:** smears predictions toward an average spectrum.
- **Seq2Seq:** feeds its own output back and falls into a loop, producing a repeating beat.
- **All models:** only magnitude is predicted, so phase is lost. Even the ground-truth reconstruction sounds buzzy and robotic; that is the best this pipeline can do.

## Differences from the paper and assumptions

**What we changed**
- Clips capped at 5 s (the paper has no cap)
- 20 extra dev sentences for dev loss
- Bidirectional LSTM encoder, as shown on the poster

**What the paper does not specify (our assumptions)**
- FIR filter: 101 taps, 8 kHz low-pass
- STFT hop: scipy default (50% overlap)
- Phoneme probability: 0.5
- Embedding sizes: 4 (NN), 1025 (Seq2Seq)
- Optimiser and loss: Adam, MSE
- SVR hyper-parameters: scikit-learn defaults
- Seq2Seq output layer: linear
- SVR receives the integer symbol matrix directly, since an embedding cannot be trained inside scikit-learn's SVR

## Getting started

The notebook was written for **Google Colab** with the dataset stored in Google Drive.

1. Download LJ Speech and place the `wavs/` folder and the metadata CSV (`id|text|normalized text` format) in your Drive.
2. Open the notebook and edit the configuration cell:
   ```python
   DATA_DIR     = "/content/drive/MyDrive"
   METADATA_CSV = f"{DATA_DIR}/text to audio .csv"
   AUDIO_DIR    = f"{DATA_DIR}/wavs"
   ```
   To run locally, set `USE_DRIVE = False` and point the paths at your local copy.
3. Install dependencies:
   ```bash
   pip install numpy scipy scikit-learn joblib soundfile nltk tensorflow matplotlib pandas
   ```
4. Run all cells. Set `QUICK_TEST = True` first for a small smoke test (20 training sentences, 3 epochs).

**Notes**
- The SVR section trains `T x 1025` separate models and takes a long time. It uses `joblib` for parallelism (`SVR_N_JOBS = -1`).
- If you run out of memory, lower `MAX_DURATION` or `N_TRAIN`.
- Text-to-symbol mapping is random, so results vary slightly between runs.

## Outputs

Generated `.wav` files are saved to `outputs_paper_replication/`:

| File | Content |
|---|---|
| `ground_truth_reconstruction.wav` | A real clip pushed through the same zero-phase postprocessing |
| `svr.wav`, `nn.wav`, `seq2seq_attention.wav` | Unseen test sentence ("The quick brown fox jumps over the lazy dog.") |
| `trainsent_*.wav` | The same models on a training sentence, plus `trainsent_ground_truth.wav` |

The notebook also plots the FIR filter before/after comparison, training loss curves, and predicted spectrograms.

## Future work
- A deeper Tacotron-style model
- More training data
- Griffin-Lim or a neural vocoder in place of zero-phase inversion

## Reference

Xiao Wang, Yahan Yang, Ye Li. *End-to-End Text to Speech Synthesis.* Stanford CS 229 Final Project, Autumn 2018.

K. Ito. *The LJ Speech Dataset.* https://keithito.com/LJ-Speech-Dataset/
