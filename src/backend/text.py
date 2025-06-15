import torch
from transformers import (
    SiglipTextModel,
    SiglipTokenizer,
)

from .config import SIGLIP_MODEL


def get_siglip_tokenizer() -> SiglipTokenizer:
    print("Loading Siglip Tokenizer...")
    siglip_tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
    print("Siglip Tokenizer loaded successfully.")
    print()
    return siglip_tokenizer


def get_siglip_text_model() -> SiglipTextModel:
    print("Loading Siglip Text Model...")
    siglip_Text_model = SiglipTextModel.from_pretrained(
        SIGLIP_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    print("Siglip Text Model loaded successfully.")
    print()
    return siglip_Text_model


def generate_text_embeddings(query, text_model, tokenizer):
    # Tokenize query
    print()
    print("Tokenizing query...")
    text_inputs = tokenizer(
        [f"This is a photo of {query}"],
        padding="max_length",
        return_tensors="pt",
    ).to("cuda")
    print("Query tokenized successfully.")

    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_model(**text_inputs)
    print("Text embeddings generated successfully.")
    text_embeddings = text_embeddings["pooler_output"]
    # Normalize embeddings to unit length
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

    return text_embeddings
