#!/usr/bin/env python
"""SAM 3 (facebook/sam3) concept-prompted segmentation server. Run inside the
vps-sam3 conda env:
    conda activate vps-sam3
    uvicorn server:app --host 0.0.0.0 --port 8004

SAM3 is mask-native (it segments, boxes are derived from the mask), so this is
the one server that actually populates Detection.mask_rle.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from schema import DetectRequest, DetectResponse, Detection  # noqa: E402

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_NAME = "sam3"
CHECKPOINT = "facebook/sam3"

app = FastAPI()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        _processor = Sam3Processor.from_pretrained(CHECKPOINT)
        _model = Sam3Model.from_pretrained(CHECKPOINT).to(_device)
        _model.eval()
    return _model, _processor


def rle_encode(mask: np.ndarray) -> str:
    """Minimal COCO-style RLE (row-major run lengths), no pycocotools dependency."""
    flat = mask.flatten(order="F").astype(np.uint8)
    diffs = np.flatnonzero(np.diff(flat, prepend=-1, append=-1) != 0)
    runs = np.diff(diffs)
    return ",".join(str(int(r)) for r in runs)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    model, processor = get_model()
    t0 = time.perf_counter()

    image = Image.open(req.image_path).convert("RGB")
    inputs = processor(images=image, text=req.prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=req.threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    detections = []
    for box, score, mask in zip(results["boxes"], results["scores"], results["masks"]):
        mask_np = mask.detach().cpu().numpy().astype(bool)
        detections.append(
            Detection(
                score=float(score),
                box_xyxy=[float(v) for v in box.tolist()],
                mask_rle=rle_encode(mask_np),
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
