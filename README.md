# NeuraX 🧠

**"Control technology with thought."**

A Brain-Computer Interface (BCI) system that translates EEG brain signals into real-time digital commands. Built for accessibility — designed to help paralyzed or physically disabled individuals interact with computers using only their thoughts.

---

## Demo

The dashboard works in two modes:

- **With API running** — full pipeline (signal → preprocess → ML → predict)
- **Without API** — local JS demo mode with synthetic signals (no setup needed)

Open `index.html` in a browser to see the dashboard immediately.

---

## Features

- Synthetic EEG signal generator (realistic delta/theta/alpha/beta/gamma bands)
- Signal preprocessing pipeline (bandpass filter, notch filter, normalization)
- Feature extraction (FFT band powers + statistical features)
- ML classifier (Random Forest + SVM, auto-selects best)
- FastAPI backend with 5 REST endpoints
- Real-time dashboard with signal visualizer and virtual cursor
- Works without real EEG hardware

---

## Project Structure

```
neurax/
├── data/
│   ├── signal_generator.py     # Synthetic EEG data generator
│   └── __init__.py
├── preprocessing/
│   ├── pipeline.py             # Filter + feature extraction
│   └── __init__.py
├── models/
│   ├── classifier.py           # ML model training + inference
│   ├── neurax_model.pkl        # Saved trained model
│   └── __init__.py
├── api/
│   └── main.py                 # FastAPI REST backend
├── frontend/
│   └── index.html              # Dashboard UI (single file)
├── notebooks/
│   └── exploration.ipynb       # Jupyter demo notebook
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python models/classifier.py
```

Output:
```
[SVM (RBF)]   Test Accuracy: 100.00%
[Random Forest] Test Accuracy: 100.00%
Best model: Random Forest — saved to models/neurax_model.pkl
```

### 3. Start the API server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the dashboard

Open `index.html` in your browser. The dashboard connects to `localhost:8000` automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/status` | Model info + uptime |
| GET | `/commands` | List available commands |
| POST | `/simulate` | Generate + predict synthetic signal |
| POST | `/predict` | Predict from raw signal array |
| GET | `/session/stats` | Session statistics |

### Example: Simulate a mental command

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"command_id": 1, "noise_level": 0.3}'
```

Response:
```json
{
  "success": true,
  "simulated_command": "cursor_left",
  "prediction": {
    "command_id": 1,
    "command": "cursor_left",
    "confidence": 0.97,
    "all_probabilities": {
      "idle": 0.01,
      "cursor_left": 0.97,
      "cursor_right": 0.01,
      "select": 0.01
    }
  }
}
```

---

## Mental Commands

| ID | Command | Brain Activity |
|----|---------|---------------|
| 0 | `idle` | Resting state — dominant alpha waves (10 Hz) |
| 1 | `cursor_left` | Left motor imagery — beta burst (20 Hz) |
| 2 | `cursor_right` | Right motor imagery — beta burst (22 Hz) |
| 3 | `select` | Mental click — P300 + gamma response |

---

## Signal Processing Pipeline

```
Raw EEG → Bandpass Filter (0.5–50 Hz) → Notch Filter (50 Hz) → Normalize
→ FFT Band Powers (delta/theta/alpha/beta/gamma)
→ Statistical Features (mean/std/var/skew/kurtosis/rms/peak)
→ 12-dimensional feature vector
→ ML Classifier → Command
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Signal Processing | Python, NumPy, SciPy |
| Machine Learning | scikit-learn (SVM + Random Forest) |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Data Format | NumPy arrays, JSON |

---

## DSA Concepts Implemented

- **Queue** — signal buffer for real-time window processing
- **Sliding Window** — 1-second overlapping EEG segments
- **Hash Map** — fast command lookup by ID
- **Stack** — prediction log history (LIFO)
- **FFT** — O(n log n) frequency domain transform

---

## With Real EEG Hardware

This project uses synthetic signals by default. To connect a real EEG device:

1. Install the device SDK (e.g., OpenBCI Python, MNE-Python for research devices)
2. Replace `generate_eeg_window()` in `data/signal_generator.py` with your hardware stream
3. Pass the real signal array to `POST /predict`
4. Retrain the model with real brain data from `POST /train` (coming soon)

Compatible headsets: OpenBCI Cyton, NeuroSky MindWave, Emotiv EPOC

---

## Ethical Considerations

- No brain data is stored permanently — session data lives in memory only
- No user identification from neural signals
- Consent is required before any real signal collection
- Built for assistive use only — not for surveillance

---

## Author

**Mani** — BS English Literature, GC University Lahore  
Project Type: Academic BCI Research + Practical Implementation

---

## License

MIT License — free to use, modify, and deploy.
