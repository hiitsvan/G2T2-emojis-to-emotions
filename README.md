# G2T2: Emojis to Emotions

A **Multimodal Cyberbullying Detection System** that combines text, facial emotion, and audio modalities to efficiently detect cyberbullying behavior in online content.

---

## 📋 Project Overview

This project implements a late fusion approach for cyberbullying detection, integrating:
- **Text Analysis** (DistilBERT fine-tuned transformer)
- **Facial Emotion Recognition** (ResNet50 on FER-2013)
- **Audio Emotion Detection** (CNN on emotion speech dataset)
- **Lexicon-based Features** (LightGBM sentiment analysis)
- **Multi-label Emotion Classification** (GoEmotions BiLSTM)

The final fusion model achieves **95.65% ROC-AUC** on the test set using a random pairing strategy that ensures genuine multimodal contribution.

---

## 📁 Repository Structure

### Notebooks Organization

The `notebooks/` folder contains both experimental iterations and the final production pipeline:

#### **Deprecated Files** (Experimental/Old Approaches):
- `02_face_emotion (sqrt class weights).ipynb` - Earlier face emotion model with square-root class weighting
- `06_fusion_initial.ipynb` - Initial fusion experiments


#### **Final Production Pipeline** (In Order):
1. **`01_text_distilbert.ipynb`** - Fine-tune DistilBERT for cyberbullying text classification
2. **`02_face_emotion (cbrt class weights).ipynb`** - Train ResNet50/MobileNetV3 for facial emotion recognition with cube-root class balancing (only ResNet50 used in the final fusion model)
3. **`03_audio_emotion_CNN.ipynb`** - Train CNN for audio emotion detection from spectrograms
4. **`04_lexicon.ipynb`** - Extract lexicon-based sentiment features using LightGBM
5. **`05_emotions.ipynb`** - Multi-label emotion classification using GoEmotions BiLSTM
6. **`08_fusion_random_pairing.ipynb`** ⭐ **FINAL MODEL** - Random pairing fusion with Logistic Regression

#### **Supporting Files**:
- `utils/` - Helper functions for data loading, preprocessing, and evaluation

---

## 🚀 How to Use

### Running the Pipeline

1. **Run notebooks sequentially** (01 → 02 → 03 → 04 → 05 → 08)
2. Each notebook saves intermediate outputs to `cache/` and `artifacts/`
3. The final fusion model (`08_fusion_random_pairing.ipynb`) loads cached predictions from earlier stages

---

## 📝 Notes

- **Initial Fusion Models**: The `Initial Fusion Model` experiments and trials are documented in the project report, explaining why certain approaches were abandoned in favor of random pairing.