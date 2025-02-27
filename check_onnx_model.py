import onnx
model_path="./checkpoints/DNN.onnx"
model=onnx.load(model_path)
input_names=[input.name for input in model.graph.input]
print("input names{}".format(input_names))
output_names=[output.name for output in model.graph.output]
print("output names{}".format(output_names))
print(f"model.ir_version ---> {model.ir_version}")
print(f"model.opset_import ---> {model.opset_import}")
print(f"model.producer_name ---> {model.producer_name}")
print(f"model.producer_version ---> {model.producer_version}")
print(f"model.domain ---> {model.domain}")
print(f"model.model_version ---> {model.model_version}")
print(f"model.doc_string ---> {model.doc_string}")
print(f"model.metadata_props ---> {model.metadata_props}")
print(f"model.training_info ---> {model.training_info}")
print(f"model.functions ---> {model.functions}")
print(f"model.graph ---> {model.graph}")
