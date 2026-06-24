import os
import json
import numpy as np
import yt_dlp
import requests
from yt_dlp.utils import match_filter_func

# Import your existing librosa extraction logic
from processor import extract_beat_features

# The final database file
OUTPUT_DB = "data/artist_profiles.json"
# A temporary folder for the scraped audio
TEMP_DIR = "data/temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


def get_trending_artists():
    """Fetches trending artists from Deezer's public charts."""
    print("\nConnecting to Deezer's public API to find trending artists...")
    url = "https://api.deezer.com/chart/0/artists"

    try:
        response = requests.get(url)
        data = response.json()

        trending_artists = []
        if "data" in data:
            for artist in data["data"]:
                trending_artists.append(artist["name"])

        print(f"-> Found {len(trending_artists)} trending artists from Deezer.")
        return trending_artists

    except Exception as e:
        print(f"Error connecting to Deezer: {e}")
        return []


def scrape_and_process_artist(artist_name: str, max_songs: int = 3):
    """Searches YouTube, downloads audio, processes it, and cleans up."""
    print(f"\n[{artist_name}] Initiating automated data gathering...")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{TEMP_DIR}/%(title)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "max_downloads": max_songs,
        "match_filter": match_filter_func("duration <= 480"),
    }

    artist_vectors = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch{max_songs}:{artist_name} audio"
            print(f"-> Downloading {max_songs} tracks from YouTube...")
            ydl.download([search_query])
        except Exception as e:
            print(f"-> Error downloading {artist_name}: {e}")

    # Process whatever files were successfully downloaded
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)

        # Ensure we only process audio files
        if not file.endswith(".mp3"):
            continue

        print(f"-> Analyzing: {file}")
        vector = extract_beat_features(file_path)

        if vector is not None:
            artist_vectors.append(vector)

        # SERVER CLEANUP: Immediately delete the MP3 so we don't hoard copyright material
        os.remove(file_path)

    # Calculate the true average style if we successfully processed songs
    if len(artist_vectors) > 0:
        true_average_vector = np.mean(artist_vectors, axis=0)
        return true_average_vector.tolist()

    return None


def build_automated_database():
    # 1. Automatically grab the trending names from Deezer
    target_artists = get_trending_artists()

    if not target_artists:
        print("No artists found. Exiting...")
        return

    # NOTE FOR TESTING: Deezer returns 50 artists.
    # Processing 50 artists * 3 songs each = 150 songs. That could take an hour to run.
    # If you just want to test if it works, uncomment the line below to only grab the top 5!
    target_artists = target_artists[:5]

    database = {}

    # 2. Feed the names into the YouTube scraper
    for artist in target_artists:
        vector = scrape_and_process_artist(artist, max_songs=3)
        if vector:
            database[artist] = vector
            print(f"-> Successfully mapped profile for {artist}")
        else:
            print(f"-> Failed to map {artist}")

    # 3. Save the mathematically accurate profiles
    with open(OUTPUT_DB, "w") as f:
        json.dump(database, f, indent=2)

    print(f"\nSuccess! Database auto-generated and saved to {OUTPUT_DB}")


if __name__ == "__main__":
    build_automated_database()
