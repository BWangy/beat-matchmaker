import json
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import StandardScaler


def build_index():
    print("Loading artist database...")

    # 1. Load the JSON database
    with open("data/artist_profiles.json", "r") as f:
        database = json.load(f)

    artist_names = list(database.keys())
    # Convert the lists of features into a 2D numpy matrix
    raw_vectors = np.array(list(database.values()), dtype="float32")

    print(f"Found {len(artist_names)} artists. Scaling features...")

    # 2. Standardize the Math (Feature Scaling)
    # This grades every feature on a curve so a high BPM doesn't overpower a tiny ZCR
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(raw_vectors)

    # 3. Save the Scaler
    # We MUST save this scaler. When a producer uploads a new beat later,
    # we have to apply these exact same "grading rules" to their beat.
    with open("vector_db/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 4. Build the FAISS Index
    # We use L2 (Euclidean) distance for standard similarity search
    dimension_size = scaled_vectors.shape[1]  # Should be 19 based on our features
    index = faiss.IndexFlatL2(dimension_size)

    # Add the scaled vectors to the index
    index.add(scaled_vectors)

    # Save the compiled index and the artist name map
    faiss.write_index(index, "vector_db/faiss_index.bin")

    with open("vector_db/artist_map.json", "w") as f:
        json.dump(artist_names, f)

    print("Success! Database compiled, scaled, and saved to vector_db/")


if __name__ == "__main__":
    build_index()
