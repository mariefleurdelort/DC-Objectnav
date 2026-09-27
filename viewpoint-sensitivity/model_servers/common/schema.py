"""Shared request/response contract for all 4 model servers.

Every server exposes the same two endpoints regardless of what's underneath:
  GET  /health -> {"status": "ok", "model": "<name>"}
  POST /detect -> DetectResponse

This is what lets the orchestrator (step 3/4) treat all 4 models identically.
Each server.py adds this file's directory to sys.path at import time (see the
`sys.path.insert` line at the top of each server) — it's pure pydantic with no
heavy deps, so sharing it doesn't compromise env isolation, and it keeps the
contract from drifting between the 4 copies.
"""
from typing import List, Optional
from pydantic import BaseModel


class DetectRequest(BaseModel):
    image_path: str          # absolute path, readable by the server process
    prompt: str               # fixed per-category text prompt (config.CATEGORY_PROMPTS)
    threshold: float = 0.3    # fixed decision threshold, shared across all 4 models


class Detection(BaseModel):
    score: float
    box_xyxy: List[float]                 # absolute pixel coords
    mask_rle: Optional[str] = None        # RLE-encoded binary mask, only for mask-capable models (SAM3)


class DetectResponse(BaseModel):
    model: str
    prompt: str
    threshold: float
    detections: List[Detection]
    latency_ms: float
