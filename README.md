# Beat Matchmaker

An automated, full-stack machine learning application that analyzes user-uploaded music beats and matches them against the mathematical sonic fingerprints of trending commercial artists.

By extracting detailed acoustic features using digital signal processing (DSP) and indexing them via Facebook AI Similarity Search (FAISS), the platform allows music producers to instantaneously find which mainstream artists best fit their production style.

---

## Project Architecture

The application is split into a modern web frontend and a high-performance machine learning backend pipeline.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION                             │
│  Deezer API (Trending Artists) ──> YouTube Search (yt-dlp Scraper)     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. DSP & INDEXING ENGINE                        │
│  Raw Audio (.mp3) ──> Librosa Feature Extraction ──> FAISS Index (.bin)│
└────────────────────────────────────┬───────────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          3. APPLICATION FLOW                           │
│  React Frontend (Drop Beat) ──> FastAPI Gateway ──> Vector Match Vector│
└────────────────────────────────────────────────────────────────────────┘

```

---

## Repository Structure

```text
├── backend/                  # FastAPI & Machine Learning Pipeline
│   ├── data/
│   │   ├── artist_profiles.json  # Aggregated 19-dimensional artist features
│   │   ├── faiss_index.bin       # Compiled binary vector search index
│   │   └── temp_audio/           # Ephemeral storage for scraper processing
│   ├── src/
│   │   ├── auto_builder.py       # Automated data pipeline & Deezer/YT scraper
│   │   ├── indexer.py            # FAISS matrix compilation & normalization
│   │   ├── processor.py          # Librosa acoustic feature extraction layer
│   │   └── main.py               # FastAPI application gateway
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Client Web Interface
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # UI elements (Drag & Drop, Artist Cards)
│   │   ├── App.jsx           # Main application view
│   │   └── index.css         # Global styling
│   └── package.json          # Node.js dependencies

```

---

## Core Components

### 1. Data Gathering & Automation (`backend/src/auto_builder.py`)

Eliminates manual audio collection by autonomously generating a localized dataset:

- Queries the **Deezer API** using pagination parameters to fetch the top 100 globally trending artists without requiring complex OAuth workflows.
- Utilizes **`yt-dlp`** to parse YouTube search results for official high-quality audio streams.
- Restricts processing overhead by applying a metadata `match_filter` that instantly rejects video files longer than **8 minutes**.
- Downloads 3 tracks per artist, processes them into acoustic vectors, and calculates an unweighted average (`np.mean`) to establish a single baseline archetypal profile for that artist.
- Automatically executes an immediate disk cleanup (`os.remove()`) to wipe raw audio files post-analysis to handle server storage constraints.

### 2. Digital Signal Processing (`backend/src/processor.py`)

Parses raw audio files (`.mp3`, `.wav`, etc.) into a **19-dimensional feature vector**:

- **Temporal Features:** Beats Per Minute (BPM) and onset strength.
- **Spectral Features:** Spectral Centroid (perceived brightness) and Spectral Flatness (tonality vs. noise).
- **Timbral Features:** 13 Mel-Frequency Cepstral Coefficients (MFCCs) tracking instrumentation textures and sonic characteristics.

### 3. Vector Index Engine (`backend/src/indexer.py`)

Compiles the structural JSON profiles into an optimized search space:

- Formats standard Python structures into strict 2D NumPy matrices (`float32`).
- Normalizes varied metric scales so high-magnitude variables (like a 140 BPM tempo) do not mathematically eclipse fractional variables (like a 0.03 spectral flatness rating).
- Loads data points into a FAISS `IndexFlatL2` matrix for microsecond-level Euclidean distance matching, serializing it to disk as `faiss_index.bin`.

### 4. Client Web Application (`frontend/`)

A responsive user interface built for fast, single-page interactions:

- Features a custom drag-and-drop file boundary enabling producers to drop local production stems directly into the viewport.
- Communicates multi-part form requests directly to the FastAPI upload gateway.
- Renders ranked similarity scores alongside matched artist profiles seamlessly using dynamic cards.

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Node.js (v18+)**
- **FFmpeg** (Required on your system path for `yt-dlp` audio conversions)

### Backend Configuration

1. Navigate to the backend directory and initialize a virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

3. Run the automated data population pipeline (grabs trending artists and downloads profiles):

```bash
python src/auto_builder.py

```

4. Compile the vector index matrix:

```bash
python src/indexer.py

```

5. Spin up the development server:

```bash
uvicorn src.main:app --reload

```

### Frontend Configuration

1. Open a new terminal instance and navigate to the frontend directory:

```bash
cd frontend

```

2. Install the node packages:

```bash
npm install

```

3. Launch the web interface client:

```bash
npm run dev

```

---

## Deployment & Production Notes

- **Audio Storage Lifecycle:** Ensure that the user-uploaded file paths on the FastAPI backend follow the same strict file-deletion workflow implemented in the `auto_builder` to prevent server disk saturation.
- **Index Rebuilding:** The `faiss_index.bin` file is static once built. To keep the app fresh with current music trends, set up a recurring cron job to run `auto_builder.py` and `indexer.py` weekly to absorb new trending charts.
