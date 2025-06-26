import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.absolute()))

from src.utils import load_embedder

if __name__ == "__main__":
    path_embeddings_pt = r"D:\Projects\dhanispix\split_siglip\weights\embeddings.pt"
    embedder = load_embedder(path_embeddings_pt)
