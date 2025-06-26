import torch
import torch.nn as nn
from ResNetEncoder import Bottleneck,ResNetEncoder
from UNetDecoder import UNetDecoder
from torchsummary import summary

class ResNet_UNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=3,backbone='resnet50',pretrained=False):
        super(ResNet_UNet, self).__init__()
        # 编码器选择
        if backbone == 'resnet50':
            # ResNet-50 使用 Bottleneck 块，blocks_num=[3, 4, 6, 3]
            self.encoder = ResNetEncoder(block=Bottleneck, blocks_num=[3, 4, 6, 3], in_channels=in_channels)
        elif backbone == 'resnet34': # 示例：如果将来想支持 ResNet34
            # ResNet-34 使用 BasicBlock 块，blocks_num=[3, 4, 6, 3]
            # self.encoder = ResNetEncoder(block=BasicBlock, blocks_num=[3, 4, 6, 3], in_channels=in_channels)
            raise NotImplementedError(f"Backbone '{backbone}' not yet implemented.")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.decoder=UNetDecoder(encoder_out_channels=self.encoder.out_channels,num_classes=num_classes)

    def forward(self,x):
        # featuers 是列表 [f0,f1,f2,f3,f4]
        encoder_out_feautres = self.encoder(x)
        decoder_out_features = self.decoder(encoder_out_feautres)
        return decoder_out_features

if __name__=='__main__':
    in_channels=3
    num_classes=3
    backbone='resnet50'
    pretrained=False
    input_height = 256
    input_width = 256
    batch_size = 2

    model = ResNet_UNet(in_channels=in_channels,num_classes=num_classes,backbone=backbone,pretrained=pretrained)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, in_channels, input_height, input_width).to(device)
    print(f"\n模拟输入张量形状: {dummy_input.shape}")

    with torch.no_grad():
        output=model(dummy_input)

    print(f"\nUNet 模型最终输出形状: {output.shape}")
    expected_output_shape = (batch_size, num_classes, input_height, input_width)
    print(f"期望输出形状: {expected_output_shape}")

    assert output.shape == expected_output_shape, \
        f"UNet 最终输出形状不匹配: 期望 {expected_output_shape}, 实际 {output.shape}"

    print("\n--- UNet 模型测试成功！最终输出维度符合预期。---")
    print(f"最终输出分辨率为 {output.shape[2]}x{output.shape[3]}，通道数为 {output.shape[1]}")




