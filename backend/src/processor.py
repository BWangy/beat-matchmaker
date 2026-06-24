import librosa
import numpy as np


def extract_beat_features(file_path: str) -> np.ndarray:
    print(f"Loading and analyzing: {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=22050)

        # --- 1. Base Features ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

        mean_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mean_zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mean_rms = np.mean(librosa.feature.rms(y=y))
        mean_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mean_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # Create an array of the first 6 features
        base_features = np.array(
            [bpm, mean_centroid, mean_zcr, mean_rms, mean_rolloff, mean_bandwidth]
        )

        # --- 2. Texture Features (MFCCs) ---
        # Extract 13 MFCCs. This returns a 2D matrix of shape (13, time_frames)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Take the mean across the time axis (axis=1) to get 13 average values
        mean_mfccs = np.mean(mfccs, axis=1)

        # --- 3. Combine into final Vector ---
        # We concatenate the 6 base features with the 13 MFCCs
        # to create a single 1D vector of length 19
        feature_vector = np.concatenate((base_features, mean_mfccs))

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == "__main__":
    sample_beat = "data/raw_audio/test_beat.wav"
    vector = extract_beat_features(sample_beat)

    if vector is not None:
        print(f"\nSuccess! Extracted a vector with {len(vector)} dimensions.")
        print(f"Raw Vector Array:\n{vector}")
