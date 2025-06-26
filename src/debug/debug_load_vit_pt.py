import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.absolute()))

from src.utils import load_vit

if __name__ == "__main__":
    path_vit_pt = r"D:\Projects\dhanispix\split_siglip\weights\vit.pt"
    vit = load_vit(path_vit_pt)
