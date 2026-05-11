import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    B, H, W, C = x.shape

    # Encoder: 4 blocks — each does (H-4) then halve
    skips = []
    for channels in [64, 128, 256, 512]:
        H, W = H - 4, W - 4   # two 3x3 valid convs (skip shape)
        skips.append((H, W))
        H, W = H // 2, W // 2  # max pool

    # Bottleneck: two 3x3 valid convs, no pooling
    H, W = H - 4, W - 4

    # Decoder: 4 blocks — upsample (×2), concat with skip, two 3x3 valid convs
    for skip_H, skip_W in reversed(skips):
        H, W = H * 2, W * 2    # up-conv doubles spatial dims
        # center-crop skip to (H, W) — doesn't change H, W
        H, W = H - 4, W - 4   # two 3x3 valid convs

    # Output: 1x1 conv — spatial dims unchanged, channels → num_classes
    return np.zeros((B, H, W, num_classes))