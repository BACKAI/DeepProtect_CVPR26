import torch


def make_lpips(network: str, device: str):
    try:
        from lpips import LPIPS
    except ImportError as exc:
        raise ImportError("Generator optimization requires the 'lpips' package.") from exc
    model = LPIPS(net=network).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def lpips_scalar(model, image_a, image_b):
    value = model(image_a, image_b)
    return value.mean()

