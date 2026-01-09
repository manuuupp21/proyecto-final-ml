from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
CHAMPION_DIR.mkdir(parents=True, exist_ok=True)


DATASET_PATH = DATA_RAW_DIR / "dataset_practica_final.csv"

# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Umbral de clasificación
DEFAULT_THRESHOLD = 0.35
