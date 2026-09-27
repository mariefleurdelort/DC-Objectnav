#!/usr/bin/env python
"""YOLO-World detection server. Run inside the vps-yoloworld conda env:
    conda activate vps-yoloworld
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from schema import DetectRequest, DetectResponse, Detection  # noqa: E402

import torch
from fastapi import FastAPI
from ultralytics import YOLO

MODEL_NAME = "yolo-world"
CHECKPOINT = "yolov8s-worldv2.pt"  # auto-downloaded by ultralytics on first use

app = FastAPI()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(CHECKPOINT)
        # set_classes()'s CLIP text encoder isn't covered by predict()'s
        # automatic device placement -- without this it stays on CPU while the
        # rest of the model runs on CUDA, raising a device-mismatch RuntimeError
        # the moment set_classes() is called.
        _model.to(_device)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    model = get_model()
    t0 = time.perf_counter()

    # YOLO-World is a fixed-vocabulary-per-call model: set the single-category
    # vocabulary to our one fixed prompt, then run standard detection.
    model.set_classes([req.prompt])
    results = model.predict(req.image_path, conf=req.threshold, device=_device, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            detections.append(
                Detection(
                    score=float(box.conf[0]),
                    box_xyxy=[float(v) for v in box.xyxy[0].tolist()],
                )
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    return DetectResponse(
        model=MODEL_NAME,
        prompt=req.prompt,
        threshold=req.threshold,
        detections=detections,
        latency_ms=latency_ms,
    )
