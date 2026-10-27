from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] # src/splendor/paths.py -> repo root

def ensure_parent(path: str | Path) -> Path:
    """Create the parent folder of `path` if needed, and return it as a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Reference
BIKENWLOOPS_DIR = Path.home() / "dev/refs/bikenwloops"

def bikenwloops_network_path(area: str) -> Path:
    return BIKENWLOOPS_DIR / "data/processed" / area / "network_preprocessed.xz"


# Data
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

def network_path(area: str) -> Path:
    return INTERIM / area / "network.gpkg"

def loops_path(area: str, max_nodes: int, max_length_m: int) -> Path:
    return PROCESSED / area / f"loops_{max_nodes}_{max_length_m}.npz"


# Basemap05 raw
BASEMAP05_DIR = RAW / "basemap_05_public_geotif"
BASEMAP05_LU01_RAW = BASEMAP05_DIR / "lu_01_2024.tif"
BASEMAP05_LU01_VAT = BASEMAP05_DIR / "lu_01_2024.tif.vat.dbf"

# Basemap05 interim (per study area)
def basemap05_lu01_path(area: str) -> Path:
    return INTERIM / area / "basemap05_lu01.tif"

def basemap05_main_types_path(area: str) -> Path:
    return INTERIM / area / "basemap05_main_types.tif"
