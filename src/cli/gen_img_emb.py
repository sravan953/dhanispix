import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())

import joblib

from backend.utils import get_images_from_directory, load_images
from backend.vision import (
    generate_image_embeddings,
    get_siglip_image_processor,
    get_siglip_vision_model,
)


def main():
    path_input_root = input(
        "Enter the path to the input image directory. "
        "Image embeddings will be saved in this directory.: "
    )
    path_input = Path(path_input_root).resolve()

    image_files = get_images_from_directory(path_input)
    input("Press Enter to continue...")

    # Load images
    images = load_images(image_files)

    # Load vision model and image processor
    vision_model = get_siglip_vision_model()
    image_processor = get_siglip_image_processor()

    # Generate image embeddings
    image_embeddings = generate_image_embeddings(images, vision_model, image_processor)

    # Save image embeddings
    print()
    print("Saving image embeddings...")
    image_embeddings = image_embeddings.cpu().numpy()
    files_to_embeddings = dict(zip(image_files, image_embeddings))
    joblib.dump(
        files_to_embeddings,
        path_input / "image_embeddings.joblib",
    )
    print(f"Image embeddings saved to {path_input / 'image_embeddings.joblib'}.")


if __name__ == "__main__":
    main()
