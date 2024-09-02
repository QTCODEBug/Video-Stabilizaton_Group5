import torch
import torch.quantization
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


model = CAIN(training=False, depth=3)  
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
#Convert the Quantized Model to ONNX
# Đầu vào giả cho mô hình CAIN
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
print('Mô hình CAIN đã được xuất sang định dạng ONNX.')

# Đầu vào giả cho mô hình ResNet
dummy_input_resnet = torch.randn(1, 15, 256, 256).cuda()  # Điều chỉnh kích thước nếu cần

# Xuất mô hình ResNet sang ONNX
torch.onnx.export(
    resnet,
    dummy_input_resnet,  # Đầu vào giả
    "resnet_model.onnx",  # Tên tệp xuất ra
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print('Mô hình ResNet đã được xuất sang định dạng ONNX.')

#trtexec --onnx=quantized_model.onnx --saveEngine=quantized_model.trt --fp16 (down to fp32->fp16)
trtexec --onnx=quantized_model.onnx --saveEngine=quantized_model.trt --fp16


