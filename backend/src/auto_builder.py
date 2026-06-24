import os
import json
import numpy as np
import yt_dlp
import requests
from yt_dlp.utils import match_filter_func
import subprocess
import shutil

# Configuration paths
OUTPUT_DB = "data/artist_profiles.json"
SEED_FILE = "data/seed_artists.txt"  # Local file for your historical/classic artists
TEMP_DIR = "data/temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Production-relevant genre mappings from Deezer
GENRE_MAP = {"Hip-Hop": 116, "Electronic": 156, "R&B": 165, "Pop": 132}


def get_multi_genre_artists(limit_per_genre: int = 50):
    """Strategy 1: Fetches top trending artists across multiple production genres."""
    trending_artists = []
    print("\nConnecting to Deezer's public API to find multi-genre trends...")

    for genre_name, genre_id in GENRE_MAP.items():
        print(f"-> Fetching trending {genre_name} artists (ID: {genre_id})...")
        url = f"https://api.deezer.com/chart/{genre_id}/artists?limit={limit_per_genre}"

        try:
            response = requests.get(url)
            data = response.json()

            genre_count = 0
            if "data" in data:
                for artist in data["data"]:
                    trending_artists.append(artist["name"])
                    genre_count += 1
            print(f"   Collected {genre_count} artists from {genre_name}.")

        except Exception as e:
            print(f"   Error fetching {genre_name} charts: {e}")

    return trending_artists


def get_seed_artists():
    """Strategy 2: Reads a static list of historical artists from a local text file."""
    print(f"\nReading historical seed list from {SEED_FILE}...")
    if not os.path.exists(SEED_FILE):
        print(f"-> Warning: {SEED_FILE} not found. Skipping historical seeds.")
        return []

    with open(SEED_FILE, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace, and ignore empty lines or comments
        artists = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    print(f"-> Found {len(artists)} seed artists from local file.")
    return artists


def scrape_and_process_artist(artist_name: str, max_songs: int = 8):
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
        "match_filter": match_filter_func("duration <= 480"),  # Skip tracks > 8 mins
    }

    artist_vectors = []

    # Deferred internal import to ensure module execution path safety
    from src.processor import extract_beat_features

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch10:{artist_name} audio"
            ydl.download([search_query])
        except Exception as e:
            print(f"-> Error downloading {artist_name}: {e}")

    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        if not file.endswith(".mp3"):
            continue

        print(f"-> Isolating Instrumental via GPU: {file}")

        # Create a dedicated output folder for Demucs
        demucs_out_dir = os.path.join(TEMP_DIR, "demucs_output")

        try:
            # We use --two-stems=vocals to tell Demucs we only want to split Vocals vs. The Beat
            subprocess.run(
                [
                    "demucs",
                    "-n",
                    "htdemucs",
                    "--two-stems=vocals",
                    "--out",
                    demucs_out_dir,
                    file_path,
                ],
                check=True,
                capture_output=True,
            )

            # Demucs nests outputs like: demucs_output/htdemucs/song_name/no_vocals.wav
            base_name = os.path.splitext(file)[0]
            instrumental_path = os.path.join(
                demucs_out_dir, "htdemucs", base_name, "no_vocals.wav"
            )

            if os.path.exists(instrumental_path):
                print(f"-> Extracting Librosa Features from Instrumental...")
                vector = extract_beat_features(instrumental_path)
                if vector is not None:
                    artist_vectors.append(vector)
            else:
                print(f"-> Error: Demucs failed to generate instrumental for {file}")

        except subprocess.CalledProcessError as e:
            # If the GPU crashes or runs out of memory, it logs the error here
            error_msg = e.stderr.decode("utf-8", errors="ignore")
            print(f"-> Demucs Pipeline Failed on {file}:\n{error_msg}")

        finally:
            # CRITICAL CLEANUP: Demucs generates massive uncompressed .wav files.
            # We must wipe both the original .mp3 and the Demucs folder before the next track.
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(demucs_out_dir):
                shutil.rmtree(demucs_out_dir)

    if len(artist_vectors) > 0:
        return np.mean(artist_vectors, axis=0).tolist()
    return None


def build_automated_database():
    # 1. Load existing database profiles to preserve previous work
    database = {}
    if os.path.exists(OUTPUT_DB):
        try:
            with open(OUTPUT_DB, "r") as f:
                database = json.load(f)
            print(f"Loaded existing database with {len(database)} profiles.")
        except Exception:
            print("Could not parse existing database. Starting fresh.")

    # 2. Gather both target pools
    live_trends = get_multi_genre_artists(limit_per_genre=50)
    historical_seeds = get_seed_artists()

    # 3. Combine both lists into a set to automatically eliminate overlapping duplicates
    combined_targets = set(live_trends + historical_seeds)
    print(
        f"\nTotal unique target artists to process across all streams: {len(combined_targets)}"
    )

    if not combined_targets:
        print("No target artists found. Exiting...")
        return

    # 4. Processing loop with duplicate checkpoints
    for artist in combined_targets:
        # Checkpoint: Skip processing if the profile was generated in a previous run
        if artist in database:
            print(f"[{artist}] Already exists in profiles database. Skipping...")
            continue

        vector = scrape_and_process_artist(artist, max_songs=8)
        if vector:
            database[artist] = vector
            print(f"-> Successfully mapped profile for {artist}")

            # Incremental save: Protect progress if script is forcefully stopped
            with open(OUTPUT_DB, "w") as f:
                json.dump(database, f, indent=2)
        else:
            print(f"-> Failed to map {artist}")

    print(
        f"\nPipeline run complete. Total profiles now in {OUTPUT_DB}: {len(database)}"
    )


if __name__ == "__main__":
    build_automated_database()
