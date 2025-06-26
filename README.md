# 📸 dhanispix

## Overview
Dhanispix is privacy-first photo search tool. 

## Features
- 🔐 **Privacy-preserving image processing:** Split-usage of SigLIP Vision Model to generate intermediate embeddings on-device and compute final image embeddings on compute-rich device
- 🔐 **Privacy-preserving text search:** SigLIP Text Model runs on-device to retrieve best-matching images so the input query never leaves your device 
- 🤸🏼‍♂️ **Flexible workflow:** Compute-rich device can be local (CUDA required) or cloud-based

## 🛠️ Installation & setup
### ⭐ Overview
Dhanispix can run in 2 modes:
1. 💻 Fully local: the frontend web UI, the SigLIP text model, and the SigLIP vision model backend run on-device
2. ☁️💻 Cloud + local: the frontend web UI and the SigLIP text model run on-device while the SigLIP vision model runs on the cloud

### ⚙️ Setup
1. Create an [ngrok](http://ngrok.com) account and copy the authentication token
2. Server backend setup:
    1. `pip install -r requirements.txt`
    2. Run `src/1a_save_embeddings_pt.py` to save the SigLIP vision model's embedder separately to disk.
    3. Run `src/1b_save_vit_py.py` to save the rest of the SigLIP vision model (encoder, post-layernorm, and head) to disk.
    4. Run `src/2_convert_onnx.py` to convert the embedder model saved in `Step 2.2` to ONNX file format.
    5. Move `.onnx` file to `app/public`.
    6. Set the path to the `.onnx` model from `Step 2.4` in the `lifespan` method of the notebook.
    7. Paste the ngrok authentication token from `Step 1` of in the `if __name__ == "__main__"` block of the notebook.
    8. Run the SigLIP server backend notebook on your local device.
    9. Note the ngrok public URL displayed in the terminal output.
3. Vite frontend setup - webapp:
    1. `cd app`
    2. `npm install`
    2. `npm run dev`
    3.  Paste the ngrok public URL from `Step 2.9`

## 📝 Todos
1. **`pan_and_scan`:** Implement [pan and scan](https://github.com/google/gemma_pytorch/blob/main/gemma/siglip_vision/pan_and_scan.py) from Gemma 3 for richer search.
2. **Compression:** Images are resized to `224x224x3` and the on-device embeddings are `196x768`. These are the same size. Investigate if the embeddings can be compressed, or if a few additional layers of the ViT can run on-device to for bandwidth savings.
3. **Save and load embeddings:** Allow user to save and load embeddings from disk for faster re-runs. The saved file should map embeddings to images and intelligently detect and process only the newer images in a future re-run.

## 🧑🏼‍💻 Contributing
Fork & PR!

