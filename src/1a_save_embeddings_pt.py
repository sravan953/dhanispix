"""Save the embedder part of SigLIP vision model as a .pt file."""

import torch
from transformers import SiglipVisionModel


def main(path_save: str):
    model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    # Get embeddings module
    modules1 = list(model.named_children())
    modules2 = list(modules1[0][1].named_children())
    embeddings = modules2[0][1]

    torch.save(embeddings.state_dict(), path_save)


if __name__ == "__main__":
    path_save = r""
    main(path_save)
