import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.absolute()))

import torch


def main(path_model_pt: str | Path):
    path_model_pt = Path(path_model_pt)
    model_name = path_model_pt.stem
    path_save_onnx = path_model_pt.with_name(f"{model_name}_onnx.onnx")

    if model_name == "embeddings":
        print("Loading embeddings model...")

        from src.utils import load_embedder

        model = load_embedder(str(path_model_pt))
        example_inputs = (torch.randn(1, 3, 224, 224).to("cuda"),)
    else:
        raise ValueError

    print("Converting to ONNX...")
    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    print("Optimizing...")
    onnx_program.optimize()
    print("Saving...")
    onnx_program.save(path_save_onnx)
    print("Done")


if __name__ == "__main__":
    # Embeddings
    path_model_pt = ""

    main(path_model_pt)
