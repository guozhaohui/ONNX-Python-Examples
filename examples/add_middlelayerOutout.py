import onnx

def main() -> None:

    model_file_path = "deploy.onnx"
    model = onnx.load(model_file_path)
    #print(model)

    inter_layers = ['gpu_0/res5_2_branch2c_bn_2', 'gpu_0/res5_1_branch2c_bn_2']

    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in inter_layers:
            print(idx, node.name)
            value_info_protos.append(node)

    model.graph.output.extend(value_info_protos)
    onnx.checker.check_model(model)
    onnx.save(model, 'deploy_debug.onnx')

if __name__ == "__main__":

    main()
