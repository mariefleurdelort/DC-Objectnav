#!/usr/bin/env python
"""OWLv2 detection server. Run inside the vps-owlv2 conda env:
    conda activate vps-owlv2
    uvicorn server:app --host 0.0.0.0 --port 8003
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from schema import DetectRequest, DetectResponse, Detection  # noqa: E402

import torch
from fastapi import FastAPI
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_NAME = "owlv2"
CHECKPOINT = "google/owlv2-base-patch16-ensemble"

app = FastAPI()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        _processor = Owlv2Processor.from_pretrained(CHECKPOINT)
        _model = Owlv2ForObjectDetection.from_pretrained(CHECKPOINT).to(_device)
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
    inputs = processor(text=[[req.prompt]], images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=_device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=req.threshold,
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
