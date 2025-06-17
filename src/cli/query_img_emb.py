import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

import joblib
import numpy as np
import torch

from backend.config import LOGIT_BIAS, LOGIT_SCALE
from backend.text import (
    generate_text_embeddings,
    get_siglip_text_model,
    get_siglip_tokenizer,
)


def main():
    path_input_root = input(
        "Enter the path to the directory containing the image embeddings. "
        "Image embeddings will be saved in this directory.: "
    )
    # path_input_root = r"C:\Users\sravan953\Pictures\Edits"

    query = input("Which images do you want to find?: ")

    # Load image embeddings
    print()
    print("Loading image embeddings...")
    files_to_embeddings = joblib.load(
        Path(path_input_root).resolve() / "image_embeddings.joblib"
    )
    print("Image embeddings loaded successfully.")
    image_files = list(files_to_embeddings.keys())
    image_embeddings = np.stack(list(files_to_embeddings.values()))
    image_embeddings = torch.from_numpy(image_embeddings).to("cuda")

    # Load text model and tokenizer
    tokenizer = get_siglip_tokenizer()
    text_model = get_siglip_text_model()

    # Generate text embeddings
    text_embeddings = generate_text_embeddings(query, text_model, tokenizer)

    # Calculate similarity
    logits_per_text = torch.matmul(text_embeddings, image_embeddings.t())
    logit_scale = torch.tensor(LOGIT_SCALE).to(logits_per_text.device)
    logit_bias = torch.tensor(LOGIT_BIAS).to(logits_per_text.device)
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
    probs = torch.sigmoid(logits_per_text)
    logits_per_text_sorted_indices = logits_per_text.argsort(dim=1, descending=True)

    print("Matching results:")
    for i, idx in enumerate(logits_per_text_sorted_indices[0]):
        print(f"{i + 1}. {image_files[idx]} - Score: {probs[0][idx].item():.4f}")


if __name__ == "__main__":
    main()
