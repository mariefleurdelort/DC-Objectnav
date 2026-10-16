#!/usr/bin/env python
"""One command for the xy right-word (true-prompt) pass on the v1 offset grids:
checks + (re)starts the 4 model servers, fixes image paths after the move into v1/,
smoke-tests every model on one real image, sets aside logs from failed runs, then runs
orchestrator/run_inference.py grid by grid and verifies rows actually succeed.
If anything fails it stops and prints the server's own error.

    cd ~/projects/DC-ObjectNav/viewpoint-sensitivity
    tmux new -s xy
    conda activate vps-orchestrator
    python xy_true_pass.py                 # all 4 grids (smallest first)
    python xy_true_pass.py --grids owlv2   # just one
    python xy_true_pass.py --check-only    # diagnose + smoke test, run nothing long

Re-running is safe: finished work is kept, failed logs are moved to v1/failed_logs/.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "v1" / "data" / "renders_framing_falsepositive"
MODELS = ["groundingdino", "owlv2", "sam3", "yoloworld"]
ENV = dict(os.environ, PYTHONNOUSERSITE="1")


def say(msg):
    print(f"[xy] {msg}", flush=True)


def run(cmd, timeout=None):
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ------------------------------------------------------------------ servers
def healthy_models():
    _, out = run([sys.executable, "orchestrator/health_check.py"], timeout=120)
    return {m for m in MODELS if re.search(rf"\[ok\]\s+{m}\b", out)}


def restart_servers(wait_s=420):
    say("restarting model servers (stop + start)...")
    run(["bash", "orchestrator/launch_servers.sh", "stop"], timeout=120)
    code, out = run(["bash", "orchestrator/launch_servers.sh", "start"], timeout=300)
    print(out.strip())
    t0 = time.time()
    while time.time() - t0 < wait_s:
        ok = healthy_models()
        if ok == set(MODELS):
            say(f"all 4 servers healthy after {time.time() - t0:.0f}s")
            return True
        time.sleep(15)
    say(f"servers still not healthy after {wait_s}s: missing {sorted(set(MODELS) - healthy_models())}")
    return False


def server_log_tail(model, n=20):
    """Newest file under logs/ (or orchestrator/) whose name mentions the model."""
    cands = []
    for d in (ROOT / "logs", ROOT / "orchestrator", ROOT / "model_servers" / model):
        if d.exists():
            cands += [p for p in d.rglob("*") if p.is_file() and model in p.name.lower()
                      and p.suffix in (".log", ".txt", ".out", ".err", "")]
    if not cands:
        return f"(no log file mentioning '{model}' found under logs/; see orchestrator/launch_servers.sh for where it writes)"
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    lines = newest.read_text(errors="replace").splitlines()
    return f"--- {newest.relative_to(ROOT)} (last {n} lines) ---\n" + "\n".join(lines[-n:])


def explain(text):
    t = text.lower()
    if "out of memory" in t:
        return ("GPU out of memory: stop other GPU jobs (LocateAnything etc., check nvidia-smi), "
                "then re-run this script.")
    if "no such file" in t or "filenotfound" in t or "cannot identify image" in t:
        return "the server cannot open the image file (path problem): paste this output to Claude."
    if "connection refused" in t:
        return "server not running: this script will restart servers on the next run."
    return "unrecognised server error: paste this output to Claude."


# ------------------------------------------------------------------ manifests
def fixed_manifest(grid):
    """Copy of framing_manifest.jsonl with rgb_path made absolute and pointing into v1/."""
    src = V1 / grid / "framing_manifest.jsonl"
    dst = V1 / grid / "framing_manifest_v1paths.jsonl"
    missing = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            p = Path(r["rgb_path"])
            if not p.is_absolute():
                for cand in (ROOT / "v1" / p, ROOT / p):
                    if cand.exists():
                        p = cand
                        break
                else:
                    p = ROOT / "v1" / p
            if not p.exists():
                missing += 1
            r["rgb_path"] = str(p)
            fo.write(json.dumps(r) + "\n")
    return dst, missing


def first_image(manifest):
    return json.loads(open(manifest).readline())["rgb_path"]


# ------------------------------------------------------------------ logs
def log_stats(path):
    rows = ok = 0
    if path.exists():
        for line in open(path):
            if line.strip():
                rows += 1
                ok += json.loads(line).get("error") is None
    return rows, ok


def set_aside_if_failed(path, grid):
    rows, ok = log_stats(path)
    if rows and ok < 0.5 * rows:
        dest = ROOT / "v1" / "failed_logs" / f"{grid}_framing_true_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)
        say(f"{grid}: previous log was mostly errors ({ok}/{rows} ok) -> moved to {dest.relative_to(ROOT)}")
    elif rows:
        say(f"{grid}: keeping existing log ({ok}/{rows} ok rows), the runner resumes it")


# ------------------------------------------------------------------ main
def smoke_test(manifest):
    img = first_image(manifest)
    bad = []
    for m in MODELS:
        code, out = run([sys.executable, "orchestrator/detect_client.py", "--model", m, "--image", img,
                         "--prompt", "a chair"], timeout=600)
        failed = code != 0 or re.search(r"Traceback|Internal Server Error|\b500\b|Connection refused", out)
        say(f"smoke test {m:13s}: {'FAILED' if failed else 'ok'}")
        if failed:
            bad.append((m, out.strip()[-800:]))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", nargs="+", default=["owlv2", "yoloworld", "sam3", "groundingdino"])
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    # 1. servers
    ok = healthy_models()
    say(f"healthy servers: {sorted(ok)}")
    if ok != set(MODELS) and not restart_servers():
        for m in sorted(set(MODELS) - healthy_models()):
            print(server_log_tail(m))
        sys.exit("[xy] stopped: servers did not come up (logs above).")

    # 2. manifests with working image paths
    manifests = {}
    for g in args.grids:
        man, missing = fixed_manifest(g)
        manifests[g] = man
        say(f"{g}: manifest with v1 image paths -> {man.relative_to(ROOT)} ({missing} images missing)")
        if missing:
            sys.exit(f"[xy] stopped: {missing} images of grid {g} not found on disk, e.g. {first_image(man)}")

    # 3. one real request per model (catches 500s before thousands of rows fail)
    bad = smoke_test(manifests[args.grids[0]])
    if bad:
        for m, out in bad:
            print(f"\n=== {m}: client output ===\n{out}\n{server_log_tail(m)}")
            say(f"-> {explain(out + server_log_tail(m))}")
        sys.exit("[xy] stopped before the long run: fix the error above, then re-run this script.")
    if args.check_only:
        say("all checks passed (--check-only, nothing run).")
        return

    # 4. run grid by grid, verifying rows succeed
    for g in args.grids:
        out_log = V1 / g / "framing_true_inference_log.jsonl"
        set_aside_if_failed(out_log, g)
        say(f"{g}: running inference (resumable; Ctrl+C is safe)...")
        code = subprocess.call([sys.executable, "orchestrator/run_inference.py", "--models", *MODELS,
                                "--manifest", str(manifests[g]), "--output", str(out_log)], cwd=ROOT, env=ENV)
        rows, okrows = log_stats(out_log)
        say(f"{g}: done (exit {code}) -> {okrows}/{rows} rows ok")
        if rows and okrows < 0.5 * rows:
            for m in MODELS:
                print(server_log_tail(m, 10))
            sys.exit(f"[xy] stopped: grid {g} mostly failed; see server logs above.")
    say("all grids finished. Next: python confusion_metrics.py --root v1 2>&1 | tee v1/confusion_summary.txt")


if __name__ == "__main__":
    main()
