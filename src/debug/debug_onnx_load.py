import onnx

onnx_model = onnx.load("D:\Projects\dhanispix\split_siglip\model.onnx")
onnx.checker.check_model(onnx_model)
