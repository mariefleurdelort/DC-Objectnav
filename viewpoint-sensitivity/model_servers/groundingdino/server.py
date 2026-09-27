#!/usr/bin/env python
"""GroundingDINO detection server. Run inside the vps-groundingdino conda env:
    conda activate vps-groundingdino
    uvicorn server:app --host 0.0.0.0 --port 8002
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from schema import DetectRequest, DetectResponse, Detection  # noqa: E402

import torch
from fastapi import FastAPI
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_NAME = "grounding-dino"
CHECKPOINT = "IDEA-Research/grounding-dino-tiny"

app = FastAPI()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(CHECKPOINT)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(CHECKPOINT).to(_device)
        _model.eval()
    return _model, _processor


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    model, processor = get_model()
    t0 = time.perf_counter()

    image = Image.open(req.image_path).convert("RGB")
    # GroundingDINO expects lowercase, period-terminated phrases.
    text = req.prompt.strip().lower()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=req.threshold,
        text_threshold=req.threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = [
        Detection(score=float(score), box_xyxy=[float(v) for v in box.tolist()])
        for box, score in zip(results["boxes"], results["scores"])
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    return DetectResponse(
        model=MODEL_NAME,
        prompt=req.prompt,
        threshold=req.threshold,
        detections=detections,
        latency_ms=latency_ms,
    )
