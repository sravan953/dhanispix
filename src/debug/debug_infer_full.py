import torch
from PIL import Image
from transformers import (
    SiglipImageProcessorFast,
    SiglipVisionModel,
)


def get_siglip_vision_model() -> SiglipVisionModel:
    print()
    print("Loading Siglip Vision Model...")
    siglip_vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    print("Siglip Vision Model loaded successfully.")
    print()
    return siglip_vision_model


def get_siglip_image_processor() -> SiglipImageProcessorFast:
    print()
    print("Loading Siglip Image Processor...")
    siglip_image_processor_fast = SiglipImageProcessorFast.from_pretrained(
        "google/siglip-base-patch16-224"
    )
    print("Siglip Image Processor loaded successfully.")
    print()
    return siglip_image_processor_fast


def generate_image_embeddings(images, vision_model, image_processor):
    # Process images
    image_files_processed = image_processor(
        images,
        return_tensors="pt",
    ).to("cuda")

    # Get image embeddings
    print("Generating image embeddings...")
    with torch.no_grad():
        image_embeddings = vision_model(**image_files_processed)
    image_embeddings = image_embeddings["pooler_output"]
    print("Image embeddings generated successfully.")
    # Normalize embeddings to unit length
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

    return image_embeddings


image_processor = get_siglip_image_processor()
vision_model = get_siglip_vision_model()
images = [Image.open(r"C:\Users\sravan953\Pictures\Edits\G0070151.jpg").convert("RGB")]
embeddings = generate_image_embeddings(images, vision_model, image_processor)

print(embeddings.shape, embeddings.mean(), embeddings.std())
