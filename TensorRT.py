import torch
import torch.quantization
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime


model = CAIN(training=False, depth=3)  
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
#Convert the Quantized Model to ONNX
# Đầu vào cho mô hình CAIN
dummy_input_cain1 = torch.randn(1, 3, 256, 256).cuda()  # Điều chỉnh kích thước nếu cần
dummy_input_cain2 = torch.randn(1, 3, 256, 256).cuda()  # Đầu vào thứ hai cho CAIN

# Xuất mô hình CAIN sang ONNX
torch.onnx.export(
    cain,
    (dummy_input_cain1, dummy_input_cain2),  # Đầu vào giả
    "cain_model.onnx",  # Tên tệp xuất ra
    opset_version=11,
    input_names=['input1', 'input2'],
    output_names=['output']
)

# Đầu vào giả cho mô hình ResNet
dummy_input_resnet = torch.randn(1, 15, 256, 256).cuda()  
# Xuất mô hình ResNet sang ONNX
torch.onnx.export(
    resnet,
    dummy_input_resnet,  # Đầu vào giả
    "resnet_model.onnx",  # Tên tệp xuất ra
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

#trtexec --onnx=quantized_model.onnx --saveEngine=quantized_model.trt --fp16 (down to fp32->fp16)
trtexec --onnx=quantized_model.onnx --saveEngine=quantized_model.trt --fp16

# Kiểm tra mô hình CAIN
ort_session_cain = onnxruntime.InferenceSession("cain_model.onnx")
outputs_cain = ort_session_cain.run(None, {'input1': dummy_input_cain1.cpu().numpy(), 'input2': dummy_input_cain2.cpu().numpy()})
print('Đầu ra mô hình CAIN từ ONNX Runtime:', outputs_cain)

# Kiểm tra mô hình ResNet
ort_session_resnet = onnxruntime.InferenceSession("resnet_model.onnx")
outputs_resnet = ort_session_resnet.run(None, {'input': dummy_input_resnet.cpu().numpy()})
print('Đầu ra mô hình ResNet từ ONNX Runtime:', outputs_resnet)



