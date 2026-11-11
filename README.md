# 🕵️‍♂️ Artisan Audio Forensics

**A Deep Learning-Based Framework for Synthetic Voice and Speech Manipulation Detection**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-green)

## 📌 Project Overview
This repository contains the code and framework for a Capstone Project focused on Audio Forensics. The system is designed to detect AI-generated (spoofed) speech versus genuine human speech. It utilizes the benchmark **ASVspoof 2019 Logical Access (LA)** dataset and compares the performance of a custom Convolutional Neural Network (CNN) against a Transformer-based architecture.

The project features a full end-to-end pipeline: from raw audio preprocessing and Mel-spectrogram generation to model training, evaluation (ROC-AUC, EER), and deployment via an interactive, custom-styled web interface.

## ✨ Key Features
*   **Audio Processing Pipeline:** Automatically resamples, cleans, and converts audio to 128x256 log-Mel spectrograms.
*   **Fast Training via Caching:** Pre-computes and caches spectrograms to `.npy` files, reducing GPU idle time and accelerating training epochs by up to 10x.
*   **Dual Deep Learning Architectures:**
    *   **CNN:** A robust 4-block 2D Convolutional Neural Network for learning local time-frequency patterns.
    *   **Transformer:** A multi-head self-attention network for capturing long-range temporal dependencies in speech.
*   **Handcrafted 3D UI:** A custom Streamlit interface built with Neumorphic design principles (soft 3D shapes, parchment textures) to provide a professional "forensic tool" feel.
*   **Dual-Model Comparison:** The UI allows users to run both the CNN and Transformer simultaneously to see if they agree or disagree on a sample.
*   **Exportable Reports:** Download the forensic analysis and probabilities as a JSON file.

## 📂 Directory Structure
```text
Audio_Forensics_Project/
│
├── data/                  # (Git-ignored) Raw datasets and processed spectrograms
├── models/                # (Git-ignored) Saved model weights (.pth)
├── outputs/               # (Git-ignored) ROC curves, confusion matrices, reports
├── app/                   # Streamlit Web Interface (UI, CSS, logic)
├── src/
│   ├── data_prep/         # Dataset parsing and CSV generation
│   ├── features/          # Spectrogram extraction and caching
│   ├── models/            # PyTorch CNN and Transformer architectures
│   ├── training/          # Training loops and config
│   └── evaluation/        # Metrics (EER, F1) and plotting
├── main.py                # Main CLI entry point
└── requirements.txt       # Python dependencies

🚀 Setup and Installation
1. Clone the Repository
bash
git clone https://github.com/YourUsername/Audio_Forensics_Project.git
cd Audio_Forensics_Project
2. Create a Virtual Environment
Windows:

powershell
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Note: For GPU acceleration, ensure you install the CUDA version of PyTorch matching your system.

bash
pip install -r requirements.txt
📊 Data Preparation
Download the ASVspoof 2019 LA dataset (LA.zip) from Edinburgh DataShare.

Extract the contents into data/raw/ASVspoof/.

Generate the training/validation/test CSVs:

bash
python -m src.data_prep.prepare_data
Precompute Spectrograms: (Highly recommended for fast training)

bash
python -m src.features.precompute_spectrograms
🏋️ Training and Evaluation
Train the models using the central CLI. The training script includes early stopping and automatically saves the best weights to the models/ folder.

Train CNN:

bash
python main.py train-cnn --epochs 30
Train Transformer:

bash
python main.py train-transformer --epochs 30
Evaluate Models:

bash
python main.py evaluate-cnn
python main.py evaluate-transformer
Evaluation generates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and EER metrics, saving Confusion Matrices and ROC curves to the outputs/ directory.

🖥️ Web Application
To launch the custom Artisan Forensics UI:

bash
streamlit run app/app.py
Upload a .wav or .flac file to view the waveform, Mel-spectrogram, and the model's prediction confidence.

⚠️ Disclaimer
This system is intended as a screening aid for audio verification. It should not be treated as conclusive proof of authenticity. Deep learning models can be susceptible to adversarial attacks and out-of-distribution generalization errors.

📚 References
Wang et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.

Tak et al. (2021). End-to-End anti-spoofing with RawNet2.

Jung et al. (2022). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Frank & Schönherr (2021). WaveFake: A Data Set to Facilitate Audio Deepfake Detection.

text

### Final step to push it to GitHub:
Once you save this updated file, run these three commands in your terminal:

```powershell
git add README.md
git commit -m "Add polished README documentation"
git push