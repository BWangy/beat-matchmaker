import pickle
import json
import faiss
import numpy as np

# Import the extraction function we built earlier
from src.processor import extract_beat_features


def match_beat(file_path: str, top_k: int = 3):
    """
    Analyzes an uploaded beat and searches the FAISS index for the closest artists.
    """
    print(f"Analyzing new beat for matches: {file_path}")

    # 1. Extract the raw features (the 19-dimension vector)
    raw_vector = extract_beat_features(file_path)
    if raw_vector is None:
        print("Failed to extract features.")
        return None

    # Scikit-learn and FAISS expect 2D matrices (like a spreadsheet with 1 row)
    # So we reshape our 1D array into a 2D array: [1, 2, 3] -> [[1, 2, 3]]
    raw_vector_2d = raw_vector.reshape(1, -1)

    # 2. Load the Database Components
    try:
        with open("vector_db/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("vector_db/artist_map.json", "r") as f:
            artist_map = json.load(f)

        index = faiss.read_index("vector_db/faiss_index.bin")

    except FileNotFoundError as e:
        print(
            f"\nError: Could not find database files. Did you run indexer.py first?\n{e}"
        )
        return None

    # 3. Scale the Math
    # We transform the new beat using the EXACT same curve as the original database
    # It must be converted to float32 because FAISS requires 32-bit floats
    scaled_vector = scaler.transform(raw_vector_2d).astype("float32")

    # 4. Search the FAISS Index
    # distances: how mathematically far the beat is from the artist (closer to 0 is better)
    # indices: the ID number of the artist in our database
    distances, indices = index.search(scaled_vector, top_k)

    # 5. Format the Results
    results = []
    print("\n" + "=" * 30)
    print("MATCH RESULTS")
    print("=" * 30)

    for i in range(top_k):
        artist_id = indices[0][i]

        # FAISS returns -1 if it runs out of items to search
        if artist_id != -1 and artist_id < len(artist_map):
            artist_name = artist_map[artist_id]
            # L2 distance: lower is better
            distance_score = float(distances[0][i])

            results.append({"artist": artist_name, "distance": distance_score})

            print(f"{i+1}. {artist_name.upper()}")
            print(f"   Distance Score: {distance_score:.4f}\n")

    return results


if __name__ == "__main__":
    # Test it with the same sample beat we used earlier
    sample_beat = "data/raw_audio/test_beat.wav"

    # We are asking for the top 3 matches
    match_beat(sample_beat, top_k=3)
