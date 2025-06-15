from pathlib import Path

from PIL import Image


def get_images_from_directory(path_input: Path):
    if not path_input.is_dir():
        raise ValueError(f"Provided path {path_input} is not a directory.")

    image_files = list(path_input.glob("*.jpg")) + list(path_input.glob("*.png"))
    if not image_files:
        raise ValueError(f"No image files found in directory {path_input}.")
    print()
    print(f"Found {len(image_files)} image files in {path_input}.")

    return image_files


def load_images(image_files) -> list[Image]:
    print("Loading images...")
    images = []
    for image_file in image_files:
        try:
            images.append(Image.open(image_file).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    print(f"Loaded {len(images)} images successfully.")
    print()

    return images
