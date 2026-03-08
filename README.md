# CausalTCC

This repository contains the official implementation of the **CausalTCC** model, a framework designed for self-supervised pre-training and downstream fine-tuning on physiological time-series data, specifically focusing on Electroencephalography (EEG) for Alzheimer's Disease (AD) detection and Human Activity Recognition (HAR).

## 📊 Datasets

The current implementation utilizes the following datasets. We specifically focus on a subset of AD EEG datasets and the standard HAR dataset.

**1. Alzheimer's Disease (AD) EEG Datasets**
*   **Included Datasets:** `AD-Auditory`, `ADFTD-RS`, `BrainLat`
*   **Role:** Divided into large-scale self-supervised pre-training sets and downstream fine-tuning sets.
*   **Download Link:** [Google Drive Link](https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX)

**2. Human Activity Recognition (HAR) Dataset**
*   **Included Dataset:** `HAR`
*   **Role:** Used for both self-supervised representation learning and semi-supervised fine-tuning.
*   **Download Link:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## ⚙️ Data Preprocessing Pipeline

For EEG datasets, we apply a rigorous preprocessing pipeline to ensure signal quality. The preprocessing steps are executed as follows:

1.  **Removal of non-EEG channels:** Discard auxiliary channels such as EOG, ECG, and specific coordinate information.
2.  **Notch and Band-Pass Filtering:** Apply a notch filter (50 Hz or 60 Hz) and a band-pass filter (0.5 Hz - 45 Hz) to suppress line noise, slow drifts, and high-frequency noise outside the scalp-recorded brain activity range.
3.  **Average Re-referencing:** Applied to reduce global noise and mitigate potential baseline shifts.
4.  **Artifact Removal (ICA):** For datasets without prior artifact rejection, we utilize Independent Component Analysis (ICA) coupled with the `ICLabel` algorithm to automatically identify and drop components related to eye blinks, muscle activity, etc.
5.  **Channel Alignment:** 
    *   *Pre-training Datasets:* Aligned to a standard 19-channel montage based on the international 10-20 system (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3/T7, C3, Cz, C4, T4/T8, T5/P7, P3, Pz, P4, T6/P8, O1, O2`). Excess channels are discarded. For different montages (e.g., Biosemi), signals are projected onto the target 19 channels using 3D coordinates.
    *   *Fine-tuning Datasets:* Raw channels are natively preserved.
6.  **Frequency Alignment:** Resample all continuous EEG signals to a uniform sampling frequency of `200 Hz`.
7.  **Multi-sampling Segmentation:** Instead of fixed resampling, signals are downsampled to multiple rates (`200 Hz, 100 Hz, 50 Hz`). Signals are then sliced into half-overlapping windows of `100, 200, or 400` timestamps (e.g., 1s, 2s, or 4s if sampling at 100 Hz). Incomplete boundary segments are discarded.
8.  **Z-Score Normalization:** Applied independently per channel dynamically *during data loading* (not modifying the preprocessed files).

---

## 🚀 Usage & Training Commands

The framework operates in a two-stage training paradigm:

### 1. Self-Supervised Pre-training
To perform task-agnostic self-supervised learning on the dataset, run:

```bash
python main.py \
    --training_mode self_supervised \
    --selected_dataset HAR \
    --experiment_description HAR \
    --run_description HAR
```

### 2. Downstream Fine-tuning
After pre-training, to fine-tune the model on a downstream task (e.g., using only a fraction of labeled data in a semi-supervised context), run:

```bash
python main.py \
    --training_mode fine_tune \
    --selected_dataset HAR \
    --experiment_description HAR \
    --run_description HAR \
    --label_percentage 0.015
```
*Note: The `--label_percentage 0.015` argument indicates that only 1.5% of the labeled data is used for fine-tuning.*

---

## 📖 Requirements

*   `Python 3.8+`
*   `PyTorch`
*   `mne` (for EEG processing)
*   `scipy`, `numpy`, `scikit-learn`
*   `pip install -r reqirements.txt`
