import torch

def extract_patches_with_location(tensor, patch_size=256, stride=192):
    """
    从图像 tensor 中提取多个 patch 及其位置信息（用于测试/重建）

    参数:
        tensor: torch.Tensor，形状 [C, H, W]
        patch_size: 每个 patch 的尺寸，默认 256
        stride: 滑动窗口步长，默认 192

    返回:
        patches: List[Tensor]，每个形状为 [C, patch_size, patch_size]
        locations: List[Tuple[int, int]]，每个 patch 的左上角坐标 (h, w)
    """
    C, H, W = tensor.shape
    patches = []
    locations = []

    # 确保边缘能被覆盖：回退一步
    hs = list(range(0, H - patch_size + 1, stride))
    ws = list(range(0, W - patch_size + 1, stride))
    if (H - patch_size) % stride != 0:
        hs.append(H - patch_size)
    if (W - patch_size) % stride != 0:
        ws.append(W - patch_size)

    for h in hs:
        for w in ws:
            patch = tensor[:, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            locations.append((h, w))

    return patches, locations


def reconstruct_from_patches(pred_patches, locations, full_shape, patch_size=256):
    C, H, W = full_shape
    result = torch.zeros((C, H, W), dtype=torch.float32)
    count = torch.zeros((1, H, W), dtype=torch.float32)

    for patch, (h, w) in zip(pred_patches, locations):
        result[:, h:h+patch_size, w:w+patch_size] += patch
        count[:, h:h+patch_size, w:w+patch_size] += 1.0

    return result / count


if __name__ == '__main__':
    # 假设输入是一张图像
    img = torch.randn(3, 1400, 800)  # [C, H, W]，比如 1 通道预测热图

    patches, locations = extract_patches_with_location(img, patch_size=256, stride=192)
    print(locations)
    new_img = reconstruct_from_patches(patches, locations, img.shape, patch_size=256)
    if torch.allclose(img, new_img, atol=1e-6):
        print("yes")
    else:
        print("not exactly equal")

    print(f"裁剪得到 {len(patches)} 个 patch")
    print(f"第一个 patch 的位置: {locations[0]}")
    print(f"第一个 patch 的 shape: {patches[0].shape}")
