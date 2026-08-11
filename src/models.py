"""
U-Net model for flood segmentation from Sentinel-1 SAR imagery.

Uses segmentation_models_pytorch to build a U-Net with a pretrained
ResNet encoder, adapted to take 2-channel SAR input (VV, VH) instead
of the standard 3-channel RGB.
"""

import segmentation_models_pytorch as smp


def build_model(encoder_name="resnet34", encoder_weights="imagenet"):
    """
    Build a U-Net for binary flood segmentation.

    Args:
        encoder_name (str): backbone architecture, e.g. 'resnet34'
        encoder_weights (str): pretrained weights source, e.g. 'imagenet'.
            Note: pretrained weights were trained on 3-channel RGB images.
            Since our SAR input has 2 channels (VV, VH), the first conv
            layer is automatically reinitialized for 2 input channels
            when in_channels != 3 (smp handles this internally).

    Returns:
        torch.nn.Module: U-Net model outputting 2 classes
            (0: not water, 1: water)
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=2,       # VV, VH bands
        classes=2,           # not water / water
        activation=None,     # raw logits, softmax/argmax applied later
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_model()
    dummy_input = torch.randn(4, 2, 512, 512)  # batch of 4 SAR chips
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expected: (4, 2, 512, 512)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")