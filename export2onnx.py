import torch
import onnx
import onnxruntime
import models
import netron
def torch2onnx(onnx_name, pth_path, model, batch_size):
    onnx_file_name = onnx_name

    model.load_state_dict(torch.load(pth_path))
    model.eval()

    dummy_input = torch.randn(batch_size, 2, 128, requires_grad = True)
    #output = model(dummy_input)
    #print(output.shape)

    torch.onnx.export(model,
                    dummy_input,
                    onnx_file_name,
                    export_params = True,
                    opset_version = 10,
                    do_constant_folding= True,
                    input_names = ['input'],
                    output_names = ['output'],
                    #dynamic_axes = {'input':{0:'batch_size'},'output':{0:'batch_size'}}
                      )#需要batch=1可以取消设置dynamic_axes

def check(model_path):
    try:
        onnx.checker.check_model(model_path)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        print("The model is valid!")

def runtime(model_path, input):
    #CPU推理模式
    ort_session = onnxruntime.InferenceSession(model_path,providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def vis(model_path):
    netron.start(model_path, browse = True)


if __name__ == '__main__':
    pth_path = "./checkpoints/5_best_model2_x.pth"
    model_path = "./checkpoints/DNN.onnx"  # onnx模型路径
    onnx_name = "./checkpoints/DNN.onnx"  # 转化目标onnx文件名
    model = models.DnnNet0()

    batch_size = 1
    input = torch.randn(batch_size, 2, 128, requires_grad=True)

    torch2onnx(onnx_name, pth_path, model, batch_size)
    check(model_path)
    output = runtime(model_path, input)  # output->list (1, 1, 7, 100)
    # print(np.array(output[0]).shape)
    vis(model_path)
