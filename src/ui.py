import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image

torch.classes.__path__ = []


# Set page config
st.set_page_config(
    page_title="Image Gallery with Embeddings", page_icon="🖼️", layout="wide"
)

st.title("🖼️ Image Gallery with Embeddings")

# ========================================
# ========== Sidebar Controls ============
# ========================================
with st.sidebar:
    st.header("🎛️ Controls")

    # File uploader
    st.subheader("📁 Upload Images")
    path_images_root = st.text_input(
        "Path to images", value="", placeholder="/path/to/images"
    )

    if (path_images_root and "path_images_root" not in st.session_state) or (
        path_images_root and st.session_state.path_images_root != path_images_root
    ):
        # Clear previous session state if path changes
        if (
            "path_images_root" in st.session_state
            and st.session_state.path_images_root != path_images_root
        ):
            for key in st.session_state.keys():
                del st.session_state[key]

        # Load images from the specified path
        path_images_root = Path(path_images_root).resolve()
        if path_images_root.exists() and path_images_root.is_dir():
            st.session_state.path_images_root = str(path_images_root)
            uploaded_files = (
                list(path_images_root.glob("*.jpg"))
                + list(path_images_root.glob("*.jpeg"))
                + list(path_images_root.glob("*.png"))
            )
            uploaded_files = [file for file in uploaded_files if file.is_file()]
            st.session_state.image_files = uploaded_files
            st.session_state.images = []
            st.success(f"✅ Found {len(uploaded_files)} images!")
        else:
            st.error("Invalid directory path. Please check and try again.")
            uploaded_files = []

    st.markdown("---")

    # Generate embeddings button
    st.subheader("🔮 Generate Embeddings")
    if "image_files" in st.session_state:
        if st.button(
            "🚀 Generate Embeddings", type="primary", use_container_width=True
        ):
            from backend.vision import (
                generate_image_embeddings,
                get_siglip_image_processor,
                get_siglip_vision_model,
            )

            # Placeholder for embedding generation logic
            with st.spinner("Generating embeddings..."):
                # Load vision model and image processor
                vision_model = get_siglip_vision_model()
                image_processor = get_siglip_image_processor()

                # Generate image embeddings
                image_embeddings = generate_image_embeddings(
                    st.session_state.images, vision_model, image_processor
                )
                st.session_state.image_embeddings = image_embeddings

                image_embeddings_npy = image_embeddings.cpu().numpy()
                filenames = [file.name for file in st.session_state.image_files]
                files_to_embeddings = dict(zip(filenames, image_embeddings_npy))
                path_save_embeddings = Path(st.session_state.path_images_root)
                joblib.dump(
                    files_to_embeddings,
                    path_save_embeddings / "image_embeddings.joblib",
                )

            st.success(
                f"✅ Generated embeddings for {len(st.session_state.image_files)} images!"
            )
    else:
        st.info("Upload images first")

    st.markdown("---")

    # Load embeddings button
    st.subheader("📥 Load Embeddings")
    path_load_embeddings = st.text_input(
        "Folder or path to embeddings file or path",
        value="",
        placeholder="/path/to/embeddings.joblib",
    )
    if path_load_embeddings and "image_embeddings" not in st.session_state:
        path_load_embeddings = Path(path_load_embeddings).resolve()
        if path_load_embeddings.exists():
            if path_load_embeddings.is_dir():
                path_images_root = path_load_embeddings
                path_load_embeddings = path_load_embeddings / "image_embeddings.joblib"
            else:
                path_images_root = path_load_embeddings.parent

            files_to_embeddings = joblib.load(path_load_embeddings)
            image_files = list(files_to_embeddings.keys())
            image_files = [path_images_root / file for file in image_files]
            image_embeddings = np.stack(list(files_to_embeddings.values()))
            image_embeddings = torch.from_numpy(image_embeddings).to("cuda")
            st.session_state.image_embeddings = image_embeddings

            st.session_state.image_files = image_files
            st.session_state.images = [
                Image.open(f).convert("RGB") for f in st.session_state.image_files
            ]
            st.success("✅ Loaded embeddings successfully!")
            st.rerun()
        else:
            st.error("Embeddings file not found. Please generate embeddings first.")

    st.markdown("---")

# ========================================
# ========== Main Content ================
# ========================================
if "image_files" not in st.session_state:
    # Show welcome message when no images are loaded
    st.markdown(
        """
    <div style="text-align: center; padding: 100px 20px;">
        <p style="font-size: 18px; color: #666;">
            Please use the sidebar to upload images and get started!
        </p>
        <div style="font-size: 48px; margin: 20px 0;">📁</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Image gallery section
# If images are loaded and embeddings are not yet generated, show the images
# If embeddings are generated, that means images are already displayed
if "image_files" in st.session_state:
    # Display image count
    st.info(f"**Images loaded:** {len(st.session_state.image_files)}")

    # Query
    query = st.text_input(
        "🔍 Search Images",
        value="",
        placeholder="Type to search images by filename...",
    )

    if query:
        if "image_embeddings" not in st.session_state:
            st.error("Please generate or load image embeddings first.")
            st.stop()

        from backend.config import LOGIT_BIAS, LOGIT_SCALE
        from backend.text import (
            generate_text_embeddings,
            get_siglip_text_model,
            get_siglip_tokenizer,
        )

        # Load text model and tokenizer
        tokenizer = get_siglip_tokenizer()
        text_model = get_siglip_text_model()

        # Generate text embeddings
        text_embeddings = generate_text_embeddings(query, text_model, tokenizer)

        # Load image embeddings from session state
        image_embeddings = st.session_state.image_embeddings

        # Calculate similarity
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t())
        logit_scale = torch.tensor(LOGIT_SCALE).to(logits_per_text.device)
        logit_bias = torch.tensor(LOGIT_BIAS).to(logits_per_text.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        probs = torch.sigmoid(logits_per_text)
        logits_per_text_sorted_indices = logits_per_text.argsort(dim=1, descending=True)

        # Filter images based on query
        matching_indices = logits_per_text_sorted_indices[0].tolist()
        st.session_state.matching_indices = matching_indices
        st.session_state.filter = True

    # Create columns for image grid
    cols_per_row = 4

    if "filter" not in st.session_state or query == "":
        image_files = st.session_state.image_files
    else:
        # Filter images based on matching indices
        image_files = [
            st.session_state.image_files[i] for i in st.session_state.matching_indices
        ]

    # Display images in a grid
    for i in range(0, len(image_files), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(image_files):
                uploaded_file = image_files[i + j]
                try:
                    # Load and display image from uploaded file
                    img = Image.open(uploaded_file).convert("RGB")
                    st.session_state.images.append(img)  # Store in session state

                    with col:
                        st.image(
                            img, caption=uploaded_file.name, use_container_width=True
                        )
                except Exception as e:
                    with col:
                        st.error(f"Error loading {uploaded_file.name}: {str(e)}")


# ========================================
# =============== Footer =================
# ========================================
st.markdown("---")
st.markdown("*Built with Streamlit* 🎈")
