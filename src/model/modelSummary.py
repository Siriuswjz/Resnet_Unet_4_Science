from torch.utils.tensorboard import SummaryWriter
from ResNet_UNet import ResNet_UNet
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_channels = 3
num_classes = 2
backbone = 'resnet50'
pretrained = False
input_height = 1400
input_width = 800
batch_size = 1

model = ResNet_UNet(in_channels=in_channels, num_classes=num_classes, backbone=backbone, pretrained=pretrained).to(device)
model.eval()
input = torch.randn(batch_size,in_channels,input_height,input_width).to(device)

# 导出为 onnx
torch.onnx.export(
    model,
    input,
    "my_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    verbose=True
)

print("模型已成功保存为 my_model.onnx")