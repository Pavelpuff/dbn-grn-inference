from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

_PRIMARY_BNF = BASE_DIR / ".venv_legacy" / "bin" / "bnf"
_FALLBACK_BNF = BASE_DIR / "venv_legacy" / "bin" / "bnf"
BNFINDER_EXEC: Final[Path] = _PRIMARY_BNF if _PRIMARY_BNF.exists() else _FALLBACK_BNF

DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
NETWORKS_DIR: Final[Path] = DATA_DIR / "networks"
DATASETS_DIR: Final[Path] = DATA_DIR / "datasets"
RESULTS_DIR: Final[Path] = DATA_DIR / "results"
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"

MODEL_BNET_FILE: Final[Path] = RAW_DIR / "model.bnet"
MODEL_NAME: Final[str] = "Mammalian_Cell_Cycle"

NETWORK_SIZES: Final[list[int]] = [5, 8, 10, 13, 16]
SCORING_FUNCTIONS: Final[list[str]] = ["MDL", "BDE"]

SEED: Final[int] = 42


def ensure_directories() -> None:
    """Creates required data and asset directories if they do not exist."""
    for directory in (RAW_DIR, NETWORKS_DIR, DATASETS_DIR, RESULTS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)