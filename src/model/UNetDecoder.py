import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,is_upblock1=False):
        """
        Args:
            in_channels (int): 来自上一解码器层（或编码器瓶颈层）的输入通道数。
            skip_channels (int): 来自对应编码器层的跳跃连接的通道数。
            out_channels (int): 本块最终的输出通道数。
        """
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.is_upblock1=is_upblock1

    def forward(self, x, skip_feature):
        # 1. 上采样 (双线性插值)
        # target_size = skip_feature.shape[2:] 获取跳跃连接特征图的高度和宽度，确保精确匹配
        # x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=True)
        if self.is_upblock1 is False:
            x = self.upconv(x)
        # 2. 跳跃连接：拼接特征图 torch.cat 在通道维度 (dim=1) 上拼接
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetDecoder(nn.Module):
    """
    U-Net 解码器部分，包含一系列的上采样块。
    """

    def __init__(self, encoder_out_channels, num_classes):
        """
        Args:
            encoder_out_channels (list): 编码器各阶段的输出通道数列表。
                                        例如：[64, 256, 512, 1024, 2048]
                                        (对应 layer0, layer1, layer2, layer3, layer4)
            num_classes (int): 最终输出的通道数（摩阻场的维度，例如 2）。
        """
        super(UNetDecoder, self).__init__()

        # 解码器从最深层开始，并逐步上采样
        # 编码器特征列表的顺序是：[stage0, stage1_layer1, stage2_layer2, stage3_layer3, stage4_layer4_bottleneck]
        # 解码器需要逆序处理这些特征，并从 features[4] (bottleneck) 开始

        # up_block4: 从 bottleneck (features[4]) 上采样，跳跃连接来自 layer3 (features[3])
        self.up_block4 = UpsampleBlock(
            in_channels=encoder_out_channels[4],  # 2048 (来自 layer4/bottleneck)
            skip_channels=encoder_out_channels[3],  # 1024 (来自 layer3)
            out_channels=encoder_out_channels[3]  # 输出通道设置为1024，与跳跃连接匹配
        )

        # up_block3: 从 up_block4 的输出上采样，跳跃连接来自 layer2 (features[2])
        self.up_block3 = UpsampleBlock(
            in_channels=encoder_out_channels[3],  # 1024 (来自上一个解码块)
            skip_channels=encoder_out_channels[2],  # 512 (来自 layer2)
            out_channels=encoder_out_channels[2]  # 512
        )

        # up_block2: 从 up_block3 的输出上采样，跳跃连接来自 layer1 (features[1])
        self.up_block2 = UpsampleBlock(
            in_channels=encoder_out_channels[2],  # 512 (来自上一个解码块)
            skip_channels=encoder_out_channels[1],  # 256 (来自 layer1)
            out_channels=encoder_out_channels[1]  # 256
        )

        # up_block1: 从 up_block2 的输出上采样，跳跃连接来自7*7卷积后的特征 (features[0])
        self.up_block1 = UpsampleBlock(
            in_channels=encoder_out_channels[1],  # 256 (来自上一个解码块)
            skip_channels=encoder_out_channels[0],  # 64 (来自 7*7 卷积)
            out_channels=encoder_out_channels[0],is_upblock1=True  # 64
        )

        # 最终上采样和输出层
        # 经过 up_block1 后，特征图分辨率已是原始输入分辨率的一半 (例如 H/2, W/2)。
        # 此时通道数是 encoder_out_channels[0] (64)。
        # 将其恢复到原始输入分辨率 (H, W)，并映射到 num_classes。
        self.final_upsample = nn.ConvTranspose2d(encoder_out_channels[0],encoder_out_channels[0],kernel_size=3,stride=2,padding=1,output_padding=1)
        self.final_conv = nn.Conv2d(encoder_out_channels[0], num_classes, kernel_size=1)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features (list): 编码器前向传播返回的特征图列表。[maxpool_out, layer1_out, layer2_out, layer3_out, layer4_out]
        """
        # 从编码器最深层特征开始 (bottleneck)
        # encoder_features[4]: H/16, W/16, 2048C (bottleneck)
        # encoder_features[3]: H/8, W/8, 1024C (skip from layer3)
        # encoder_features[2]: H/4, W/4, 512C (skip from layer2)
        # encoder_features[1]: H/2, W/2, 256C (skip from layer1)
        # encoder_features[0]: H/2, W/2, 64C (skip from conv1)

        dec_x = encoder_features[4]  # 编码器最深层特征

        # 解码器路径
        dec_x = self.up_block4(dec_x, encoder_features[3])  # Output: H/16, W/16, 1024C
        dec_x = self.up_block3(dec_x, encoder_features[2])  # Output: H/8, W/8, 512C
        dec_x = self.up_block2(dec_x, encoder_features[1])  # Output: H/4, W/4, 256C
        dec_x = self.up_block1(dec_x, encoder_features[0])  # Output: H/2, W/2, 64C

        # 最终上采样到原始分辨率
        dec_x = self.final_upsample(dec_x)  # Output: H, W, 64C

        # 最终卷积映射到所需输出通道数
        output = self.final_conv(dec_x)  # Output: H, W, num_classes

        return output

if __name__ == '__main__':
    print("--- 正在测试 UNetDecoder ---")

    batch_size = 2
    num_classes = 3
    input_height = 256
    input_width = 256


    encoder_output_channels = [64, 256, 512, 1024, 2048]

    # 尺寸和通道数应与 ResNetEncoder 的实际输出匹配
    # encoder_features[0]: (64, 700, 400)
    # encoder_features[1]: (256, 700, 400)
    # encoder_features[2]: (512, 350, 200)
    # encoder_features[3]: (1024, 175, 100)
    # encoder_features[4]: (2048, 88, 50) <-- 解码器起始输入

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型已部署到: {device}")

    simulated_encoder_features = [
        torch.randn(batch_size, 64, 128, 128).to(device),
        torch.randn(batch_size, 256, 128, 128).to(device),
        torch.randn(batch_size, 512, 64, 64).to(device),
        torch.randn(batch_size, 1024, 32, 32).to(device),
        torch.randn(batch_size, 2048, 16, 16).to(device)
    ]

    print("\n模拟编码器特征图的形状:")
    for i, feature in enumerate(simulated_encoder_features):
        print(f"  features[{i}]: {feature.shape}")

    decoder = UNetDecoder(encoder_out_channels=encoder_output_channels, num_classes=num_classes)
    decoder.to(device)

    with torch.no_grad():
        decoder_output = decoder(simulated_encoder_features)

    print(f"\n解码器最终输出形状: {decoder_output.shape}")

    expected_output_shape = (batch_size, num_classes, input_height, input_width)
    print(f"期望输出形状: {expected_output_shape}")

    assert decoder_output.shape == expected_output_shape, \
        f"解码器输出形状不匹配: 期望 {expected_output_shape}, 实际 {decoder_output.shape}"

    print("\n--- UNetDecoder 测试成功！最终输出维度符合预期。---")
    print(f"最终输出分辨率为 {decoder_output.shape[2]}x{decoder_output.shape[3]}，通道数为 {decoder_output.shape[1]}")
