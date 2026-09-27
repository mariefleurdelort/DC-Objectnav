"""Dataset-generation config for the controlled multi-view render.

Rendering setup follows "Active Open-Vocabulary Recognition" (Chen et al.,
https://arxiv.org/abs/2311.17938): ShapeNetCore CAD models rendered on a bare
habitat-sim stage from a discretized viewing sphere (they use 30-degree
azimuth/elevation increments, 3.0m max camera distance). Elevation is kept as
the primary swept variable for this sensitivity analysis -- each object
instance x each elevation x each model -- with a small fixed azimuth set as a
controlled nuisance variable, mirroring rendering/config.py's original design.

Edit this file to change what gets rendered; render_multiview.py just consumes it.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAPENET_ROOT = PROJECT_ROOT / "data" / "shapenet"
OBJAVERSE_ROOT = PROJECT_ROOT / "data" / "objaverse"
RENDERS_DIR = PROJECT_ROOT / "data" / "renders"

# Which 3D object source render_multiview.py pulls from: "shapenet" or
# "objaverse". ShapeNetCore's HF dataset (scripts/download_shapenet.sh) is
# gated and needs a license approval that can take a while; Objaverse
# (scripts/download_objaverse.sh) needs no login at all, so it's the default
# until that approval comes through. Both dataset backends produce the exact
# same RenderableObject interface (model_id, category, glb_path), so switching
# this string is the only change needed elsewhere.
DATASET_SOURCE = "objaverse"  # "shapenet" | "objaverse"

# category -> ShapeNetCore v2 synset id. All 8 are valid ShapeNetCore55 synsets
# and were chosen to overlap with the indoor/furniture vocabulary the 4
# detection model servers already use (model_servers/*/server.py).
SHAPENET_SYNSETS = {
    "chair": "03001627",
    "table": "04379243",
    "sofa": "04256520",
    "cabinet": "02933112",
    "bed": "02818832",
    "bookshelf": "02871439",
    "lamp": "03636649",
    "bathtub": "02808440",
}

# category -> LVIS-annotation synonym(s) to match against Objaverse's LVIS
# category keys (see rendering/objaverse_objects.py:find_matching_lvis_keys).
# LVIS uses "bookcase" rather than "bookshelf", hence the mismatch there.
OBJAVERSE_CATEGORY_KEYWORDS = {
    "chair": ["chair"],
    "table": ["table"],
    "sofa": ["sofa", "couch"],
    "cabinet": ["cabinet"],
    "bed": ["bed"],
    "bookshelf": ["bookcase"],
    "lamp": ["lamp"],
    "bathtub": ["bathtub"],
}

CATEGORIES = list(SHAPENET_SYNSETS.keys())
assert CATEGORIES == list(OBJAVERSE_CATEGORY_KEYWORDS.keys()), \
    "SHAPENET_SYNSETS and OBJAVERSE_CATEGORY_KEYWORDS must define the same categories, same order"

# One fixed text prompt per category, reused identically across every model and
# every view (this is what makes elevation the only thing being varied).
CATEGORY_PROMPTS = {
    "chair": "a chair",
    "table": "a table",
    "sofa": "a sofa",
    "cabinet": "a cabinet",
    "bed": "a bed",
    "bookshelf": "a bookshelf",
    "lamp": "a lamp",
    "bathtub": "a bathtub",
}

# How many model instances to sample per category, for whichever DATASET_SOURCE
# is active (None = all downloaded).
MAX_MODELS_PER_CATEGORY = 20

# Elevation angles in degrees, measured from the object's horizontal plane
# (0 = camera at object height looking straight on, positive = looking down from
# above, negative = looking up from below). This is the primary independent
# variable. The source paper uses 30-degree increments over a full viewing
# sphere (M=12 azimuths x N=12 elevations, including redundant/mirrored angles
# from wrapping elevation through 360 degrees); we instead use a physically
# non-redundant dome covering "below" to "directly above", still in 30-degree steps.
ELEVATIONS_DEG = [-60, -30, 0, 30, 60, 90]

# Azimuth angles in degrees around the object (0 = canonical "front", increasing
# counter-clockwise viewed from above). Matches the paper's 30-degree/12-step
# azimuth discretization exactly.
AZIMUTHS_DEG = list(range(0, 360, 30))

# Distance from camera to object center, as a multiple of the object's own AABB
# diagonal, so the object roughly fills a similar fraction of frame regardless of
# its absolute size. The paper instead uses a fixed max distance of 3.0m; we scale
# by object size since ShapeNet models span a wide range of real-world scales
# (a lamp vs. a bed) and a single fixed distance would over/under-frame many of them.
ORBIT_RADIUS_SCALE = 1.8
MIN_ORBIT_RADIUS_M = 0.75
MAX_ORBIT_RADIUS_M = 3.0  # caps at the paper's max camera distance

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
HFOV_DEG = 90

# category -> the single "visually confusable" wrong category tested against
# it in the false-positive experiment (orchestrator/run_inference.py --false-
# positive). Picked by hand for plausible visual similarity; where no strong
# match exists among our 8 categories (table, lamp, bathtub), the closest
# available one was chosen anyway rather than skipping the category.
FALSE_CATEGORY_MAP = {
    "chair": "sofa",
    "sofa": "chair",
    "cabinet": "bookshelf",
    "bookshelf": "cabinet",
    "bed": "sofa",
    "table": "cabinet",
    "lamp": "bookshelf",
    "bathtub": "bed",
}
assert set(FALSE_CATEGORY_MAP.keys()) == set(CATEGORIES), "FALSE_CATEGORY_MAP must cover every category"

MANIFEST_PATH = RENDERS_DIR / "manifest.jsonl"

# Minimum fraction of the object's silhouette that must be visible (not
# self-occluded / not clipped by frustum), relative to that instance's own
# best (least-occluded) sampled view, for a view to be kept.
MIN_VISIBLE_FRACTION = 0.3
