import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

import base64
import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import (
    SiglipImageProcessorFast,
    SiglipTextModel,
    SiglipTokenizer,
    SiglipVisionModel,
)

siglip_image_processor: SiglipImageProcessorFast = None
siglip_vision_model: SiglipVisionModel = None
siglip_tokenizer: SiglipTokenizer = None
siglip_text_model: SiglipTextModel = None
device: str = "cuda"
images: list[Image] = []
images_b64_for_react: list[dict[str, str]] = []
image_embeddings: torch.Tensor = []
text_embeddings: torch.Tensor = []
LOGIT_SCALE = 4.765625
LOGIT_BIAS = -12.9296875


class SearchQuery(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.text import get_siglip_text_model, get_siglip_tokenizer
    from backend.vision import get_siglip_image_processor, get_siglip_vision_model

    global \
        siglip_image_processor, \
        siglip_vision_model, \
        siglip_tokenizer, \
        siglip_text_model, \
        device, \
        images, \
        images_b64_for_react, \
        embeddings
    device = "cuda"
    siglip_image_processor = get_siglip_image_processor()
    siglip_vision_model = get_siglip_vision_model()
    siglip_vision_model.to(device)
    siglip_vision_model.eval()
    siglip_tokenizer = get_siglip_tokenizer()
    siglip_text_model = get_siglip_text_model()
    siglip_text_model.to(device)
    siglip_text_model.eval()
    yield


app = FastAPI(title="SigLIP Image Search API", lifespan=lifespan)


# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/clear-variables/")
async def clear_variables():
    """Reset the service state"""
    global images, images_b64_for_react, embeddings
    images = []
    images_b64_for_react = []
    embeddings = []
    return {"status": "Service state reset successfully."}


@app.post("/upload-images/")
async def upload_images(files: list[UploadFile] = File(...)):
    """Process uploaded images"""
    global images, images_b64_for_react

    for file in files:
        content = await file.read()
        bytes = io.BytesIO(content)

        # PIL
        img = Image.open(bytes).convert("RGB")
        images.append(img)

        # Convert to base64 for React
        img_str = base64.b64encode(bytes.getvalue()).decode()
        img_str = f"data:image/jpeg;base64,{img_str}"
        img_obj = {
            "id": file.filename,
            "filename": file.filename,
            "base64_data": img_str,
        }
        images_b64_for_react.append(img_obj)

    return {"images": images_b64_for_react, "total": len(images)}


@app.post("/generate-embeddings/")
async def generate_embeddings():
    """Generate embeddings for the uploaded images"""
    global images, images_b64_for_react, image_embeddings

    print(f"Generating embeddings for {len(images)} images...")
    images_processed = siglip_image_processor(images, return_tensors="pt").to(device)
    print(f"Images processed: {len(images_processed)}")
    print()

    print("Generating embeddings...")
    with torch.no_grad():
        image_embeddings = siglip_vision_model(**images_processed)
    image_embeddings = image_embeddings["pooler_output"]
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
    print(f"Embeddings generated: {image_embeddings.shape}")
    print()

    return {}


@app.post("/search/")
async def search(query: SearchQuery):
    """Search for images based on a text query"""
    global \
        image_embeddings, \
        images_b64_for_react, \
        siglip_tokenizer, \
        siglip_text_model, \
        device, \
        LOGIT_SCALE, \
        LOGIT_BIAS

    # Tokenize the request
    request_tokens = siglip_tokenizer(
        [f"This is a photo of {query}"],
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    # Get the text embeddings
    with torch.no_grad():
        text_embeddings = siglip_text_model(**request_tokens)
    text_embeddings = text_embeddings["pooler_output"]
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

    # Calculate cosine similarity
    logits_per_text = torch.matmul(text_embeddings, image_embeddings.t())
    logit_scale = torch.tensor(LOGIT_SCALE).to(logits_per_text.device)
    logit_bias = torch.tensor(LOGIT_BIAS).to(logits_per_text.device)
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

    # Sort best matches
    best_indices = torch.argsort(logits_per_text, descending=True)
    best_indices = best_indices[:10].cpu().detach().numpy().flatten()
    best_images = [images_b64_for_react[i] for i in best_indices]

    return {
        "results": best_images,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn src.service:app --reload
