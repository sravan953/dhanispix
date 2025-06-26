import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.absolute()))

import torch
from PIL import Image
from transformers import (
    SiglipImageProcessorFast,
)

from split_siglip.load_embeddings_pt import load_embeddings
from split_siglip.load_vit_pt import load_vit

path_embeddings_pt = r"D:\Projects\dhanispix\split_siglip\weights\embeddings.pt"
path_vit_pt = r"D:\Projects\dhanispix\split_siglip\weights\vit.pt"

image_processor = SiglipImageProcessorFast.from_pretrained(
    "google/siglip-base-patch16-224"
)
images = [Image.open(r"C:\Users\sravan953\Pictures\Edits\G0070151.jpg").convert("RGB")]
image_files_processed = image_processor(
    images,
    return_tensors="pt",
).to("cuda")
embeddings_model = load_embeddings(path_embeddings_pt)
remaining_model = load_vit(path_vit_pt)
with torch.no_grad():
    embeddings_output = embeddings_model(**image_files_processed)
    embeddings = remaining_model(embeddings_output)
embeddings = embeddings["pooler_output"]
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
