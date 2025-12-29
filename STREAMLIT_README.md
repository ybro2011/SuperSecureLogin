# Streamlit Web Application - Celebrity Look-Alike Challenge

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the app:**
   - The app will open in your default browser
   - Click "START" to begin the webcam stream
   - Allow camera access when prompted

## Features

- **Real-time Face Matching**: Uses face_recognition library to match your face with celebrities
- **Live Similarity Score**: Real-time percentage showing how well you match the celebrity
- **Heatmap Wireframe**: Visual overlay showing green (good match) and red (mismatch) areas
- **Registration**: Add new faces to the database directly from the webcam
- **Cached Loading**: Celebrity database is cached for fast startup

## How It Works

- The app uses `streamlit-webrtc` to handle webcam feeds in the browser
- Each frame is processed by the `VideoTransformer` class
- Face recognition runs on downscaled images (25% size) for performance
- MediaPipe Face Mesh provides 468-point facial landmark detection
- The heatmap shows where your face shape differs from the matched celebrity

## Notes

- After registering a new face, you may need to refresh the webcam stream to see it in matching
- The `encodings.pickle` cache file is automatically updated when new faces are added
- The app uses Streamlit's caching to prevent reloading the celebrity database on every frame

