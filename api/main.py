"""
NeuraX - FastAPI Backend
Endpoints:
  GET  /              - health check
  GET  /status        - system status
  POST /predict       - predict command from raw signal array
  POST /simulate      - generate + predict a synthetic signal
  GET  /commands      - list all available commands
  GET  /session/stats - session statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.classifier import load_model, predict, COMMANDS
from data.signal_generator import generate_eeg_window, WINDOW_SAMPLES
from preprocessing.pipeline import extract_features

# ─── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuraX API",
    description="Brain-Computer Interface signal processing and command prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────────────────────

_model = None
_session_start = time.time()
_prediction_log = []  # stores last 100 predictions


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


# ─── Schemas ─────────────────────────────────────────────────────────────────

class SignalInput(BaseModel):
    signal: list[float] = Field(
        ...,
        description=f"Raw EEG signal array of exactly {WINDOW_SAMPLES} float values",
        min_length=WINDOW_SAMPLES,
        max_length=WINDOW_SAMPLES,
    )


class SimulateInput(BaseModel):
    command_id: Optional[int] = Field(
        None,
        description="Force a specific command (0=idle, 1=cursor_left, 2=cursor_right, 3=select). Random if omitted.",
        ge=0,
        le=3,
    )
    noise_level: float = Field(0.3, description="Signal noise level (0.0 - 1.0)", ge=0.0, le=1.0)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project": "NeuraX",
        "tagline": "Control technology with thought.",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/status")
def status():
    model = get_model()
    uptime = round(time.time() - _session_start, 1)
    return {
        "model_loaded": model is not None,
        "model_type": type(model.named_steps["clf"]).__name__,
        "uptime_seconds": uptime,
        "total_predictions": len(_prediction_log),
        "commands_available": len(COMMANDS),
        "sample_rate_hz": 256,
        "window_samples": WINDOW_SAMPLES,
    }


@app.get("/commands")
def get_commands():
    return {
        "commands": [
            {"id": k, "name": v, "description": _command_descriptions()[v]}
            for k, v in COMMANDS.items()
        ]
    }


@app.post("/predict")
def predict_from_signal(body: SignalInput):
    """
    Accept a raw EEG signal array and return the predicted mental command.
    """
    model = get_model()
    signal = np.array(body.signal, dtype=np.float32)

    try:
        result = predict(signal, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log prediction
    entry = {
        "timestamp": time.time(),
        "command_id": result["command_id"],
        "command": result["command"],
        "confidence": result["confidence"],
    }
    _prediction_log.append(entry)
    if len(_prediction_log) > 100:
        _prediction_log.pop(0)

    return {
        "success": True,
        "prediction": result,
        "signal_length": len(body.signal),
    }


@app.post("/simulate")
def simulate_and_predict(body: SimulateInput):
    """
    Generate a synthetic EEG signal and predict its command.
    Useful for testing and frontend demo.
    """
    model = get_model()

    cmd_id = body.command_id
    if cmd_id is None:
        cmd_id = int(np.random.choice(list(COMMANDS.keys())))

    signal = generate_eeg_window(command=cmd_id, noise_level=body.noise_level)
    result = predict(signal, model=model)

    # Log prediction
    _prediction_log.append({
        "timestamp": time.time(),
        "command_id": result["command_id"],
        "command": result["command"],
        "confidence": result["confidence"],
    })
    if len(_prediction_log) > 100:
        _prediction_log.pop(0)

    return {
        "success": True,
        "simulated_command_id": cmd_id,
        "simulated_command": COMMANDS[cmd_id],
        "signal_preview": signal[:32].tolist(),  # first 32 samples for chart
        "full_signal": signal.tolist(),
        "prediction": result,
    }


@app.get("/session/stats")
def session_stats():
    if not _prediction_log:
        return {"message": "No predictions yet", "total": 0}

    commands_seen = [p["command"] for p in _prediction_log]
    avg_confidence = sum(p["confidence"] for p in _prediction_log) / len(_prediction_log)

    command_counts = {}
    for cmd in commands_seen:
        command_counts[cmd] = command_counts.get(cmd, 0) + 1

    return {
        "total_predictions": len(_prediction_log),
        "average_confidence": round(avg_confidence * 100, 2),
        "command_distribution": command_counts,
        "recent_predictions": _prediction_log[-10:],
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _command_descriptions():
    return {
        "idle":         "No active mental command detected. Brain is in resting state.",
        "cursor_left":  "Left motor imagery - moves cursor or selection to the left.",
        "cursor_right": "Right motor imagery - moves cursor or selection to the right.",
        "select":       "Mental click / P300 response - confirms current selection.",
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
