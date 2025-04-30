from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR        = "/home/eda/Desktop/dynamic/EchoNet-Dynamic"
OUTPUT_DIR      = BASE_DIR / "output"
MODEL_NAME      = "deeplabv3_resnet50"
WEIGHTS_PATH    = OUTPUT_DIR / "segmentation" / "deeplabv3_resnet50_pretrained" / "best.pt"