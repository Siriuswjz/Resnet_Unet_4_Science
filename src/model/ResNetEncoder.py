import torch.nn as nn
import torch



class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 第一个1x1卷积：压缩通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 3x3卷积：进行主要的空间特征提取。注意：这里的stride决定了下采样！
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 第三个1x1卷积：恢复（或扩展）通道数
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNetEncoder(nn.Module):
    def __init__(self, block, blocks_num, in_channels=3,
                 groups=1, width_per_group=64):
        """
        Args:
            block: 使用的残差块类型 (BasicBlock 或 Bottleneck)。
            blocks_num (list): 每个 layer 的残差块数量，例如 [3, 4, 6, 3] 用于 ResNet-50。
            in_channels (int): 输入图像的通道数，例如 3 (RGB) 或你的速度场通道数。
            groups (int): 卷积组数（用于 ResNeXt）。
            width_per_group (int): 每组的宽度。
        """
        super(ResNetEncoder, self).__init__()
        self.in_channel = 64 # ResNet 初始卷积层的输出通道数

        self.groups = groups
        self.width_per_group = width_per_group

        # Initial Stem
        self.conv1 = nn.Conv2d(in_channels, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # ResNet Layers
        # _make_layer 函数会根据 block.expansion 自动调整通道数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 用于存储每个阶段的输出通道数，方便 U-Net 解码器使用
        # 对应： Stem (conv1+maxpool), layer1, layer2, layer3, layer4
        if block == Bottleneck:
            self.out_channels = [64, 256, 512, 1024, 2048] # 具体到Bottleneck，这些是最终的输出通道数
        else:
            raise ValueError("Unsupported block type.")

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        # layer 1 channel=64,block_num=3 stride=1 self.in_channel=64 downsample=(64,256) return:(256)
        # layer 2 self.in_channel=256,channel= 128,block_num=4 stride=2 downsample=(256,512) return:(512)
        downsample = None
        # 如果需要下采样或通道数不匹配，则添加下采样层
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        # first block (256,128) downsample!=None,stride=2 return (128*4)
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion # 更新当前输入通道数

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # feature_stem = x # H/2, W/2, 64C (conv1 after relu, before maxpool)

        # x = self.maxpool(x)
        features.append(x) # stage0: H/2, W/2, 64C

        # Layer 1
        x = self.layer1(x)
        features.append(x) # stage1: H/2, W/2, 256C (for Bottleneck)

        # Layer 2
        x = self.layer2(x)
        features.append(x) # stage2: H/4, W/4, 512C

        # Layer 3
        x = self.layer3(x)
        features.append(x) # stage3: H/8, W/8, 1024C

        # Layer 4 (Bottleneck of the encoder)
        x = self.layer4(x)
        features.append(x) # stage4: H/16, W/16, 2048C

        return features

# def resnet50():
#     # https://download.pytorch.org/models/resnet50-19c8e357.pth
#     return ResNetEncoder(Bottleneck, [3, 4, 6, 3])
#
#
#
# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return ResNetEncoder(Bottleneck, [3, 4, 6, 3],
#                   groups=groups,
#                   width_per_group=width_per_group)


if __name__ == '__main__':
    print("--- 正在测试 ResNetEncoder ---")
    # 1. 定义测试参数
    input_channels = 3
    input_height = 1400

    input_width = 800
    batch_size = 2  # 使用一个批次大小为2的输入

    # 2. 实例化 ResNetEncoder
    encoder = ResNetEncoder(block=Bottleneck, blocks_num=[3, 4, 6, 3], in_channels=input_channels)

    # 3. 将模型移到可用设备 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    print(f"模型已部署到: {device}")

    # 4. 创建一个模拟输入张量
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width).to(device)
    print(f"\n模拟输入张量形状: {dummy_input.shape}")

    # 5. 执行前向传播并获取特征图
    with torch.no_grad():  # 测试时不需要计算梯度
        output_features = encoder(dummy_input)

    # 6. 验证输出特征图的数量
    expected_num_features = 5  # ResNetEncoder 应该返回 5 个特征图 (stage0 到 stage4)
    assert len(output_features) == expected_num_features, \
        f"期望 {expected_num_features} 个特征图，但实际得到 {len(output_features)} 个。"
    print(f"\n特征图数量检查通过: 得到了 {len(output_features)} 个特征图。")

    # 7. 验证每个特征图的维度和通道数
    print("\n--- 验证每个特征图的维度 ---")

    expected_dims = [
        (64, 700, 400),  # features[0]: Conv1 + MaxPool (stride=1)
        (256, 700, 400),  # features[1]: Layer1
        (512, 350, 200),  # features[2]: Layer2
        (1024, 175, 100),  # features[3]: Layer3
        (2048, 88, 50)  # features[4]: Layer4 (Bottleneck)
    ]

    for i, feature_map in enumerate(output_features):
        expected_c, expected_h, expected_w = expected_dims[i]

        print(f"特征图 {i}:")
        print(f"  实际形状: {feature_map.shape}")
        print(f"  期望形状: (BatchSize, {expected_c}, {expected_h}, {expected_w})")

        assert feature_map.shape[0] == batch_size, f"Batch size mismatch for feature map {i}."
        assert feature_map.shape[
                   1] == expected_c, f"通道数不匹配 for feature map {i}: 期望 {expected_c}, 实际 {feature_map.shape[1]}."
        assert feature_map.shape[
                   2] == expected_h, f"高度不匹配 for feature map {i}: 期望 {expected_h}, 实际 {feature_map.shape[2]}."
        assert feature_map.shape[
                   3] == expected_w, f"宽度不匹配 for feature map {i}: 期望 {expected_w}, 实际 {feature_map.shape[3]}."
        print(f"  特征图 {i} 维度检查通过。")

    print("\n--- ResNetEncoder 测试成功！所有特征图维度均符合预期。---")
