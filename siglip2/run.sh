python export.py
python benchmark.py

onnxsim conv_embeddings.onnx conv_embeddings-sim.onnx
onnxcli draw -t svg conv_embeddings-sim.onnx svgs/conv_embeddings-sim.svg
onnxsim linear_embeddings.onnx linear_embeddings-sim.onnx
onnxcli draw -t svg linear_embeddings-sim.onnx svgs/linear_embeddings-sim.svg
onnxsim linear_embeddings_image_input.onnx linear_embeddings_image_input-sim.onnx
onnxcli draw -t svg linear_embeddings_image_input-sim.onnx svgs/linear_embeddings_image_input-sim.svg
