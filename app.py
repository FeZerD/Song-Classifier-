import streamlit as st
import joblib
import librosa
import numpy as np

# --- Helper Functions ---

@st.cache_resource
def load_assets():
    """Loads the pre-trained model and other essential files from their full paths."""
    try:
        # --- UPDATED with your specific file paths ---
        # The 'r' before the string is important; it tells Python to treat backslashes as literal characters.
        model = joblib.load(r"C:\Users\SCAIPL\Desktop\Music Classifier\Data\genre_classifier_model.pkl")
        scaler = joblib.load(r"C:\Users\SCAIPL\Desktop\Music Classifier\Data\scaler.pkl")
        label_encoder = joblib.load(r"C:\Users\SCAIPL\Desktop\Music Classifier\Data\label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Model or other asset files not found. Please double-check the file paths in the script.")
        return None, None, None

def extract_features_from_audio(file):
    """Extracts all 58 features from an audio file to match the model's training data."""
    try:
        y, sr = librosa.load(file, mono=True, duration=30)
        
        # --- Feature Extraction ---
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        y_harm, y_perc = librosa.effects.hpss(y)
        tempo = librosa.feature.tempo(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # --- Combine all features in the correct order ---
        features = [
            len(y),
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr),
            np.mean(y_harm), np.var(y_harm),
            np.mean(y_perc), np.var(y_perc),
            np.mean(tempo)
        ]
        
        # Add all MFCCs
        for mfcc in mfccs:
            features.append(np.mean(mfcc))
            features.append(np.var(mfcc))
            
        return np.array(features)

    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
        return None

# --- Streamlit App Interface ---

st.set_page_config(page_title="Music Genre Classifier", layout="wide")

st.title("🎵 Music Genre Classifier")
st.markdown("Upload a **.wav** audio file, and the model will predict its genre.")

model, scaler, label_encoder = load_assets()

if model is not None:
    uploaded_file = st.file_uploader("Choose a .wav file...", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Classify Genre"):
            with st.spinner('Analyzing the song...'):
                features = extract_features_from_audio(uploaded_file)
                
                if features is not None:
                    if len(features) != scaler.n_features_in_:
                        st.error(f"Feature mismatch! Extracted {len(features)} features, but model expects {scaler.n_features_in_}. Please check the feature extraction logic.")
                    else:
                        features = features.reshape(1, -1)
                        features_scaled = scaler.transform(features)
                        prediction_numerical = model.predict(features_scaled)
                        prediction_genre = label_encoder.inverse_transform(prediction_numerical)
                        
                        st.success(f"Predicted Genre: **{prediction_genre[0].upper()}**")

st.markdown("---")
st.write("Built with Streamlit and Scikit-learn.")