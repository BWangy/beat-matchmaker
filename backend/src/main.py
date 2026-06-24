from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# Import the matcher function we just built
from src.matcher import match_beat

app = FastAPI(title="Beat Matchmaker API")

# --- CORS Configuration ---
# This is mandatory. It tells the browser that our upcoming React frontend
# (which runs on port 5173) is allowed to talk to this backend (port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the temporary audio folder exists
UPLOAD_DIR = "data/raw_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "Engine is running smoothly."}


@app.post("/api/match")
async def upload_and_match(file: UploadFile = File(...)):
    """
    Receives an audio file, saves it temporarily, finds the best artist matches,
    and then deletes the audio file.
    """
    # 1. Validate the file type (basic security check)
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail="File must be an audio file (mp3, wav, etc.)"
        )

    # 2. Save the file temporarily to our disk
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Pass the saved file to our FAISS matcher engine
        matches = match_beat(temp_file_path, top_k=3)

        if matches is None:
            raise HTTPException(
                status_code=500, detail="Failed to analyze the audio file."
            )

        # 4. Return the results to the frontend
        return {"filename": file.filename, "matches": matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5. SERVER CLEANUP: Always delete the file after processing
        # This runs even if the code above crashes, preventing disk space exhaustion
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
