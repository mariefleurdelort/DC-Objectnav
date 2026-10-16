**How many false positives a different viewpoint fixes (bench, strict = FP gone AND object still found [presence])**

| Model | False positives | Persistent (no view fixes) | Fixed ≤30° (strict) | Fixed ≤60° (strict) | Fixed ≤90° (strict) | Random 30° move fixes | Fixed ≤90° (loose) |
|---|---|---|---|---|---|---|---|
| GroundingDINO | 13729 | 100% | 0% | 0% | 1% | 0% | 1% |
| OWLv2 | 152 | 14% | 13% | 30% | 51% | 9% | 100% |
| YOLO-World | 312 | 0% | 12% | 32% | 50% | 8% | 100% |
| LocateAnything | 3107 | 34% | 24% | 55% | 67% | 19% | 90% |
| Rex-Omni | 9553 | 75% | 16% | 41% | 55% | 11% | 67% |
| CLIP | 765 | 11% | 19% | 54% | 68% | 15% | 100% |
