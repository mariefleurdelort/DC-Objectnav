#Dataset-generation config for the controlled multi-view render.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJAVERSE_ROOT = PROJECT_ROOT / "data" / "objaverse"
RENDERS_DIR = PROJECT_ROOT / "data" / "renders"

DATASET_SOURCE = "objaverse" 

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

CATEGORIES = list(OBJAVERSE_CATEGORY_KEYWORDS.keys())

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

# How many model instances to sample per category
MAX_MODELS_PER_CATEGORY = 20

# Elevation angles in degrees, measured from the object's horizontal plane
# (0 = camera at object height looking straight on, positive = looking down from
# above, negative = looking up from below). In 30 degree steps
ELEVATIONS_DEG = [-60, -30, 0, 30, 60, 90]

# Azimuth angles in degrees around the object (0 = canonical "front", increasing
# counter-clockwise viewed from above). 
AZIMUTHS_DEG = list(range(0, 360, 30))

# Distance from camera to object center
ORBIT_RADIUS_SCALE = 1.8
MIN_ORBIT_RADIUS_M = 0.75
MAX_ORBIT_RADIUS_M = 3.0  # caps at the paper's max camera distance

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
HFOV_DEG = 90

# category -> the single "visually confusable" wrong category tested against
# it in the false-positive experiment. Picked by hand for plausible visual similarity; where no strong
# match exists among our 8 categories (table, lamp, bathtub)
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

# Minimum fraction of the object's silhouette that must be visible
MIN_VISIBLE_FRACTION = 0.3
