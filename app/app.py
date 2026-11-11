import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import tempfile
import os
import json
from styles import CUSTOM_CSS
from audio_analyzer import analyze_audio
from visualizations import plot_waveform, plot_spectrogram_image

st.set_page_config(page_title="Artisan Audio Forensics", page_icon="🕵️‍♂️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🕵️‍♂️ Artisan Audio Forensics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #6b5e4f;'>A handcrafted deep learning framework for detecting synthetic voice and speech manipulation.</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar
with st.sidebar:
    st.markdown("<h2>⚙️ System Settings</h2>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Detection Engine", 
        ["compare", "cnn", "transformer"], 
        format_func=lambda x: "Compare Both (CNN vs Trans)" if x == "compare" else x.upper()
    )
    
    st.markdown("""
    <div class='3d-card' style='margin-top: 30px;'>
        <h4 style='margin-top:0;'>📌 System Info</h4>
        <p style='font-size: 0.9rem; color: #6b5e4f;'>Engine: PyTorch (CUDA)<br>
        Dataset: ASVspoof 2019 LA<br>
        Input: Mel-Spectrograms (128x256)</p>
    </div>
    """, unsafe_allow_html=True)

# Uploader
st.markdown("<div class='3d-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop an audio recording here (.wav, .flac, .mp3)", type=["wav", "flac", "mp3", "ogg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Save uploaded file to temp
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Extracting acoustic traces and running neural networks..."):
            data = analyze_audio(tmp_path, model_choice=model_choice)

        # --- 1. Audio Player & Metadata ---
        st.markdown("<h2 style='text-align: center; margin-top: 30px;'>🎧 Evidence Playback</h2>", unsafe_allow_html=True)
        col_audio, col_meta = st.columns([2, 1])
        
        with col_audio:
            st.audio(uploaded_file, format='audio/wav')
            
        with col_meta:
            meta = data["metadata"]
            st.markdown(f"""
            <div class='3d-card' style='padding: 15px;'>
                <p style='margin: 0;'><b>Duration:</b> {meta['duration_sec']} sec</p>
                <p style='margin: 0;'><b>Sample Rate:</b> {meta['sample_rate']} Hz</p>
                <p style='margin: 0;'><b>File Size:</b> {round(uploaded_file.size / 1024, 1)} KB</p>
            </div>
            """, unsafe_allow_html=True)

        # --- 2. Predictions ---
        st.markdown("<h2 style='text-align: center; margin-top: 30px;'>📊 Forensic Analysis</h2>", unsafe_allow_html=True)
        
        preds = data["predictions"]
        
        # Function to render a single model's result
        def render_result(model_name, pred_data):
            label = pred_data["label"]
            conf = pred_data["confidence"]
            color_class = "result-real" if label == "Real" else "result-fake"
            icon = "✅" if label == "Real" else "❌"
            
            st.markdown(f"""
            <div class='3d-card' style='text-align: center;'>
                <h3 style='margin-top: 0;'>{model_name}</h3>
                <div class='result-box {color_class}' style='font-size: 24px; padding: 10px;'>
                    {icon} {label}
                </div>
                <p style='font-size: 1.1rem; margin-top: 10px;'><b>Confidence:</b> {conf:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        # Display Results based on selection
        if model_choice == "compare":
            col1, col2 = st.columns(2)
            with col1:
                render_result("CNN Engine", preds["cnn"])
            with col2:
                render_result("Transformer Engine", preds["transformer"])
        else:
            render_result(model_choice.upper(), preds[model_choice])

        # --- 3. Export Report ---
        report = {
            "file_name": uploaded_file.name,
            "metadata": data["metadata"],
            "predictions": data["predictions"]
        }
        st.download_button(
            label="📥 Download Forensic JSON Report",
            data=json.dumps(report, indent=4),
            file_name=f"forensics_report_{uploaded_file.name}.json",
            mime="application/json",
            help="Download the raw predictions and metadata for your records."
        )

        st.write("---")

        # --- 4. Visualizations ---
        st.markdown("<h2 style='text-align: center;'>🖼️ Spectrogram & Waveform</h2>", unsafe_allow_html=True)
        col_wave, col_spec = st.columns(2)
        
        with col_wave:
            st.markdown("<div class='3d-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Waveform (Time Domain)</h4>", unsafe_allow_html=True)
            fig_wave = plot_waveform(data["audio"], data["sample_rate"])
            st.pyplot(fig_wave)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_spec:
            st.markdown("<div class='3d-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Mel-Spectrogram (Frequency Domain)</h4>", unsafe_allow_html=True)
            fig_spec = plot_spectrogram_image(data["mel_spectrogram"], data["sample_rate"])
            st.pyplot(fig_spec)
            st.markdown("</div>", unsafe_allow_html=True)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; opacity: 0.6;'>
        <h3>Awaiting audio sample...</h3>
        <p>Upload a file above to begin forensic analysis.</p>
    </div>
    """, unsafe_allow_html=True)