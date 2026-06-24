# Beat Matchmaker

An automated, full-stack machine learning application that analyzes user-uploaded music beats and matches them against the mathematical sonic fingerprints of trending and historical commercial artists.

By isolating instrumentals using Meta's **Demucs AI** (GPU-accelerated), extracting acoustic features via digital signal processing (**librosa**), and indexing them via Facebook AI Similarity Search (**FAISS**), the platform allows music producers to instantaneously find which mainstream artists best fit their production style.

---

## Project Architecture

The application is split into a modern web frontend and a high-performance machine learning backend pipeline heavily optimized for audio processing.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION                              │
│  Deezer API (Multi-Genre) + Local Seed List ──> yt-dlp (YouTube Audio)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        2. AI & DSP PROCESSING                           │
│  Raw MP3 ──> Demucs (GPU Stem Separation) ──> Librosa ──> FAISS Index   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          3. APPLICATION FLOW                            │
│  React Frontend (Drop Beat) ──> FastAPI Gateway ──> Vector Match Search │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## Repository Structure

```text
├── backend/                  # FastAPI & Machine Learning Pipeline
│   ├── data/
│   │   ├── artist_profiles.json  # Aggregated 19-dimensional artist features
│   │   ├── seed_artists.txt      # Static list of historical/classic artists
│   │   ├── faiss_index.bin       # Compiled binary vector search index
│   │   └── temp_audio/           # Ephemeral storage for scraper & Demucs
│   ├── src/
│   │   ├── auto_builder.py       # Scraper, Demucs controller & pipeline runner
│   │   ├── indexer.py            # FAISS matrix compilation & normalization
│   │   ├── processor.py          # Librosa acoustic feature extraction layer
|   |   ├── matcher.py            # Loads FAISS index and executes nearest-neighbor search
│   │   └── main.py               # FastAPI application gateway
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Client Web Interface
│   ├── src/
│   │   ├── components/       # UI elements (Drag & Drop, Artist Cards)
│   │   └── App.jsx           # Main application view
│   └── package.json          # Node.js dependencies

```

---

## Core Components

### 1. Hybrid Data Gathering (`auto_builder.py`)

Eliminates manual audio collection by autonomously generating a localized dataset using a dual-strategy approach:

- **Multi-Genre Live Trends:** Queries the Deezer API to fetch currently trending artists across Hip-Hop, Electronic, R&B, and Pop charts.
- **Historical Seeds:** Reads `data/seed_artists.txt` to ensure legendary and foundational producers are included in the dataset.
- **Fault Tolerance:** Saves profile math incrementally after _every_ successful artist. Features an idempotency checkpoint to automatically skip previously processed artists on subsequent runs.

### 2. GPU-Accelerated Stem Separation (Demucs)

To prevent human vocals from distorting the acoustic math, the backend utilizes **Meta's htdemucs neural network** via PyTorch:

- Runs as a `subprocess` to ensure GPU VRAM is completely flushed between tracks.
- Extracts the `no_vocals.wav` (instrumental) track, ensuring the mathematical fingerprint is based strictly on kicks, snares, 808s, and instruments.
- Triggers an immediate, aggressive server cleanup (`os.remove()`, `shutil.rmtree()`) to wipe massive uncompressed WAV files and copyrighted MP3s post-analysis.

### 3. Digital Signal Processing (`processor.py`)

Parses the isolated instrumental files into a **19-dimensional feature vector** using `librosa`:

- **Temporal Features:** Beats Per Minute (BPM) and onset strength.
- **Spectral Features:** Spectral Centroid (perceived brightness) and Spectral Flatness (tonality vs. noise).
- **Timbral Features:** 13 Mel-Frequency Cepstral Coefficients (MFCCs) tracking instrumentation textures.

### 4. Vector Index Engine (`indexer.py`)

Compiles the structural JSON profiles into an optimized FAISS `IndexFlatL2` matrix for microsecond-level Euclidean distance matching, serialized to disk as `faiss_index.bin`.

### 4. Vector Query Engine (`matcher.py`)

Acts as the search interface between the incoming user API request and the compiled matrix. It loads `faiss_index.bin` into memory, normalizes the 19-dimensional user vector, and executes a real-time nearest-neighbor search to return the top artist matches alongside calculated confidence scores.

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Node.js (v18+)**
- **FFmpeg** (Required on your system path for audio conversions)
- **NVIDIA GPU** (Required for practical Demucs processing times)

### Backend Configuration

1. Navigate to the backend directory and initialize a virtual environment:

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # On Mac/Linux: source venv/bin/activate

```

2. Install standard dependencies:

```bash
pip install -r requirements.txt

```

3. **Install CUDA-enabled PyTorch and Demucs:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install demucs

```

4. Populate your historical seed list in `data/seed_artists.txt`.
5. Run the automated data population pipeline _(Note: The first run will pause briefly to download the `htdemucs` model weights)_:

```bash
python -m src.auto_builder

```

6. Compile the vector index matrix:

```bash
python -m src.indexer

```

7. Spin up the FastAPI development server:

```bash
uvicorn src.main:app --reload

```

### Frontend Configuration

1. Open a new terminal instance and navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev

```
