import json, sys, os, glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

def crop_to_content(img, pad=40):
    """Crop image to the non-white, non-black bounding box + padding."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find pixels that are neither pure white nor pure black
    mask = (gray > 10) & (gray < 250)
    if not mask.any():
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - pad)
    r1 = min(img.shape[0], r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(img.shape[1], c1 + pad)
    return img[r0:r1, c0:c1]

def plot(json_path):
    d        = json.load(open(json_path))
    ep_id    = d["episode_id"]
    scene    = os.path.basename(d.get("scene_id", "?")).replace(".glb", "")
    outcome  = d.get("outcome", "?")
    category = d.get("object_category", "?")
    positions= d["positions"]
    base     = json_path.replace(".json", "")

    obs_path = base + "_obstacle_map.png"
    val_path = base + "_value_map.png"
    has_obs  = os.path.exists(obs_path)
    has_val  = os.path.exists(val_path)

    ncols = 1 + int(has_obs) + int(has_val)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 9))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")
    col = 0

    # ── Trajectory ────────────────────────────────────────────────────────
    ax = axes[col]; col += 1
    ax.set_facecolor("#16213e")
    xs = [p[0] for p in positions]
    zs = [p[2] for p in positions]
    n  = len(xs)

    # Color trajectory dark→bright to show time progression
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        c = (1-t) * np.array([0.05, 0.2, 0.5]) + t * np.array([0.1, 0.7, 1.0])
        ax.plot([xs[i], xs[i+1]], [zs[i], zs[i+1]], color=c, linewidth=2.5)

    ax.scatter(xs[0],  zs[0],  c="#00ff88", s=250, zorder=6,
               edgecolors="white", linewidths=1.5, label="Start", marker="o")
    ax.scatter(xs[-1], zs[-1], c="#ff4757", s=250, zorder=6,
               edgecolors="white", linewidths=1.5, label="End",   marker="X")

    # Direction arrows every ~10% of trajectory
    step = max(1, n // 10)
    for i in range(0, n - step, step):
        dx, dz = xs[i+step] - xs[i], zs[i+step] - zs[i]
        if abs(dx) + abs(dz) > 0.01:
            ax.annotate("", xy=(xs[i+step], zs[i+step]), xytext=(xs[i], zs[i]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.2, alpha=0.6))

    # Minimum 6m × 6m view with padding
    xc, zc = np.mean(xs), np.mean(zs)
    xr = max((max(xs) - min(xs)) / 2, 3.0) * 1.4
    zr = max((max(zs) - min(zs)) / 2, 3.0) * 1.4
    ax.set_xlim(xc - xr, xc + xr)
    ax.set_ylim(zc - zr, zc + zr)

    unique = len(set(zip([round(x, 2) for x in xs], [round(z, 2) for z in zs])))
    ax.set_title(f"GPS Trajectory\n{n} steps · {unique} unique positions", color="white", fontsize=12, pad=10)
    ax.set_xlabel("X (m)", color="#aaa", fontsize=11)
    ax.set_ylabel("Z (m)", color="#aaa", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, color="#3a3a6a")
    ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=10)

    # ── Obstacle + Frontier map (cropped to explored region) ─────────────
    if has_obs:
        ax = axes[col]; col += 1
        raw = cv2.cvtColor(cv2.imread(obs_path), cv2.COLOR_BGR2RGB)
        img = crop_to_content(raw, pad=60)
        ax.imshow(img)
        ax.set_title("Obstacle + Frontier Map\n(explored region)", color="white", fontsize=12, pad=10)
        ax.axis("off")

    # ── Value map (cropped to explored region) ────────────────────────────
    if has_val:
        ax = axes[col]; col += 1
        raw = cv2.cvtColor(cv2.imread(val_path), cv2.COLOR_BGR2RGB)
        img = crop_to_content(raw, pad=60)
        ax.imshow(img)
        ax.set_title(f"Value Map · target: '{category}'\ninferno = higher ITM score",
                     color="white", fontsize=12, pad=10)
        ax.axis("off")

    title_color = "#7ed321" if outcome == "success" else "#ff4757"
    fig.suptitle(
        f"Episode {ep_id}  ·  {scene}  ·  {outcome.upper()}  ·  {n} steps",
        color=title_color, fontsize=14, y=1.02
    )
    plt.tight_layout()
    out = base + "_viz.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()

target = sys.argv[1] if len(sys.argv) > 1 else "data/trajectories"
files  = sorted(glob.glob(os.path.join(target, "episode_*.json"))) \
         if os.path.isdir(target) else [target]
files  = [f for f in files if "_map" not in f]
print(f"Found {len(files)} trajectory files")
[plot(f) for f in files]
