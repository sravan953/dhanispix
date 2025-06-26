"""Save the ViT part of SigLIP vision model as a .pt file.
This script extracts the encoder, post-layernorm, and head from the SiglipVisionModel"""

import torch
from transformers import SiglipVisionModel


def main(path_save: str):
    model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    # Get remaining modules (except embeddings)
    modules1 = list(model.named_children())
    modules2 = list(modules1[0][1].named_children())
    remaining = modules2[1:]
    encoder, post_layernorm, head = remaining
    encoder, post_layernorm, head = encoder[1], post_layernorm[1], head[1]

    weights = {
        "encoder": encoder.state_dict(),
        "post_layernorm": post_layernorm.state_dict(),
        "head": head.state_dict(),
    }

    torch.save(weights, path_save.format("vit.pt"))


if __name__ == "__main__":
    path_save = ""
    main(path_save)
