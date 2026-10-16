**How many false positives a different viewpoint fixes (main, strict = FP gone AND object still found [presence])**

| Model | False positives | Persistent (no view fixes) | Fixed ≤30° (strict) | Fixed ≤60° (strict) | Fixed ≤90° (strict) | Random 30° move fixes | Fixed ≤90° (loose) |
|---|---|---|---|---|---|---|---|
| GroundingDINO | 9631 | 100% | 0% | 0% | 1% | 0% | 2% |
| OWLv2 | 82 | 0% | 56% | 85% | 99% | 25% | 100% |
| SAM3 | 928 | 21% | 26% | 50% | 58% | 14% | 97% |
| YOLO-World | 479 | 0% | 19% | 40% | 58% | 8% | 100% |
| LocateAnything | 2612 | 36% | 30% | 62% | 73% | 18% | 95% |
| Rex-Omni | 7883 | 88% | 14% | 37% | 48% | 7% | 63% |
| CLIP | 786 | 0% | 35% | 74% | 90% | 19% | 100% |
