from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_MIDI_DIR = PROJECT_ROOT / "data/raw_midi/maestro-v3.0.0-midi/maestro-v3.0.0"
METADATA_CSV = PROJECT_ROOT / "data/raw_midi/maestro-v3.0.0.csv"

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
SPLIT_DIR = PROJECT_ROOT / "data/train_test_split"

OUTPUT_MIDI_DIR = PROJECT_ROOT / "outputs/generated_midis"
PLOTS_DIR = PROJECT_ROOT / "outputs/plots"

FS = 16
SEQ_LEN = 128
STRIDE = 64
MIN_PITCH = 21
MAX_PITCH = 108

BATCH_SIZE = 16
HIDDEN_DIM = 256
LATENT_DIM = 64
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 20