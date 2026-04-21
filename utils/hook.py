import os
import argparse
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import embed_net


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_transform(img_h: int = 384, img_w: int = 144):
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_tensor(img_path: str, img_h: int = 384, img_w: int = 144) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    transform = build_transform(img_h, img_w)
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def denormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    elif x.dim() == 3:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    else:
        raise ValueError("Unsupported tensor shape for denormalize_tensor.")

    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    return x


def tensor_to_rgb_image(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(1, 2, 0).numpy()


def tensor_to_gray_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    gray = np.mean(x, axis=0)
    gray = np.clip(gray, 0.0, 1.0)
    return gray


def feature_to_response_map(
    feat: torch.Tensor,
    out_h: int,
    out_w: int,
    reduce_type: str = "abs_mean"
) -> np.ndarray:
    if reduce_type == "abs_mean":
        response = feat.abs().mean(dim=0)
    elif reduce_type == "l2":
        response = torch.norm(feat, p=2, dim=0)
    else:
        raise ValueError(f"Unsupported reduce_type: {reduce_type}")

    response = (response - response.min()) / (response.max() - response.min() + 1e-8)

    response = F.interpolate(
        response.unsqueeze(0).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    response = response.detach().cpu().numpy()
    response = np.clip(response, 0.0, 1.0)
    return response


def save_single_image(img, save_path: str, cmap: Optional[str] = None):
    plt.figure(figsize=(3, 6))
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_triplet_figure(
    base_img: np.ndarray,
    response_map: np.ndarray,
    save_path: str,
    title_prefix: str = "AFF",
    cmap_base: Optional[str] = None
):
    plt.figure(figsize=(10, 3.5))

    plt.subplot(1, 3, 1)
    if cmap_base is None:
        plt.imshow(base_img)
    else:
        plt.imshow(base_img, cmap=cmap_base)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(response_map, cmap="jet")
    plt.title(f"{title_prefix} Response")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if cmap_base is None:
        plt.imshow(base_img)
    else:
        plt.imshow(base_img, cmap=cmap_base)
    plt.imshow(response_map, cmap="jet", alpha=0.4)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_figure(
    vis_gray: Optional[np.ndarray],
    vis_response: Optional[np.ndarray],
    ir_gray: Optional[np.ndarray],
    ir_response: Optional[np.ndarray],
    save_path: str,
    title_prefix: str = "AFF"
):
    rows = 0
    if vis_gray is not None:
        rows += 1
    if ir_gray is not None:
        rows += 1

    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    row_idx = 0
    if vis_gray is not None:
        axes[row_idx, 0].imshow(vis_gray, cmap="gray")
        axes[row_idx, 0].set_title("VIS Input")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(vis_response, cmap="jet")
        axes[row_idx, 1].set_title(f"VIS {title_prefix} Response")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(vis_gray, cmap="gray")
        axes[row_idx, 2].imshow(vis_response, cmap="jet", alpha=0.4)
        axes[row_idx, 2].set_title("VIS Overlay")
        axes[row_idx, 2].axis("off")

        row_idx += 1

    if ir_gray is not None:
        axes[row_idx, 0].imshow(ir_gray, cmap="gray")
        axes[row_idx, 0].set_title("IR Input")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(ir_response, cmap="jet")
        axes[row_idx, 1].set_title(f"IR {title_prefix} Response")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(ir_gray, cmap="gray")
        axes[row_idx, 2].imshow(ir_response, cmap="jet", alpha=0.4)
        axes[row_idx, 2].set_title("IR Overlay")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_checkpoint_safely(model, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint

    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    missing = [k for k in model_dict.keys() if k not in filtered_dict]
    unexpected = [k for k in state_dict.keys() if k not in model_dict]

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"Matched params: {len(filtered_dict)}")
    if len(missing) > 0:
        print(f"Missing params (ignored): {len(missing)}")
    if len(unexpected) > 0:
        print(f"Unexpected params (ignored): {len(unexpected)}")


def visualize_aff_responses_with_hook(
    model,
    vis_path: Optional[str] = None,
    ir_path: Optional[str] = None,
    save_dir: str = "aff_response_results",
    img_h: int = 384,
    img_w: int = 144,
    hook_after: str = "conv1",   # "conv1" or "relu"
    reduce_type: str = "abs_mean",
    device: Optional[str] = None,
):
    ensure_dir(save_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    core_model = model.module if hasattr(model, "module") else model
    hook_store: Dict[str, torch.Tensor] = {}

    def save_visible_hook(module, inp, out):
        hook_store["visible_feat"] = out.detach()

    def save_thermal_hook(module, inp, out):
        hook_store["thermal_feat"] = out.detach()

    if hook_after == "conv1":
        visible_hook_layer = core_model.visible_module.visible.conv1
        thermal_hook_layer = core_model.thermal_module.thermal.conv1
        title_prefix = "AFF"
    elif hook_after == "relu":
        visible_hook_layer = core_model.visible_module.visible.relu
        thermal_hook_layer = core_model.thermal_module.thermal.relu
        title_prefix = "AFF-based Stem"
    else:
        raise ValueError("hook_after must be 'conv1' or 'relu'")

    vis_handle = visible_hook_layer.register_forward_hook(save_visible_hook)
    ir_handle = thermal_hook_layer.register_forward_hook(save_thermal_hook)

    vis_gray = None
    ir_gray = None
    vis_response = None
    ir_response = None

    try:
        if vis_path is not None:
            if not os.path.isfile(vis_path):
                raise FileNotFoundError(f"VIS image not found: {vis_path}")

            x_vis = load_image_tensor(vis_path, img_h=img_h, img_w=img_w).to(device)

            with torch.no_grad():
                _ = core_model.visible_module(x_vis)

            if "visible_feat" not in hook_store:
                raise RuntimeError("Visible hook did not capture any feature.")

            vis_feat = hook_store["visible_feat"][0]
            vis_denorm = denormalize_tensor(x_vis[0])
            vis_rgb = tensor_to_rgb_image(vis_denorm)
            vis_gray = tensor_to_gray_image(vis_denorm)

            vis_response = feature_to_response_map(
                vis_feat, out_h=img_h, out_w=img_w, reduce_type=reduce_type
            )

            save_single_image(vis_rgb, os.path.join(save_dir, "vis_input.png"))
            save_single_image(vis_response, os.path.join(save_dir, "vis_aff_response.png"), cmap="jet")
            save_triplet_figure(
                base_img=vis_gray,
                response_map=vis_response,
                save_path=os.path.join(save_dir, "vis_aff_triplet.png"),
                title_prefix=title_prefix,
                cmap_base="gray"
            )

        if ir_path is not None:
            if not os.path.isfile(ir_path):
                raise FileNotFoundError(f"IR image not found: {ir_path}")

            x_ir = load_image_tensor(ir_path, img_h=img_h, img_w=img_w).to(device)

            with torch.no_grad():
                _ = core_model.thermal_module(x_ir)

            if "thermal_feat" not in hook_store:
                raise RuntimeError("Thermal hook did not capture any feature.")

            ir_feat = hook_store["thermal_feat"][0]
            ir_denorm = denormalize_tensor(x_ir[0])
            ir_rgb = tensor_to_rgb_image(ir_denorm)
            ir_gray = tensor_to_gray_image(ir_denorm)

            ir_response = feature_to_response_map(
                ir_feat, out_h=img_h, out_w=img_w, reduce_type=reduce_type
            )

            save_single_image(ir_rgb, os.path.join(save_dir, "ir_input.png"))
            save_single_image(ir_response, os.path.join(save_dir, "ir_aff_response.png"), cmap="jet")
            save_triplet_figure(
                base_img=ir_gray,
                response_map=ir_response,
                save_path=os.path.join(save_dir, "ir_aff_triplet.png"),
                title_prefix=title_prefix,
                cmap_base="gray"
            )

        save_summary_figure(
            vis_gray=vis_gray,
            vis_response=vis_response,
            ir_gray=ir_gray,
            ir_response=ir_response,
            save_path=os.path.join(save_dir, "aff_response_summary.png"),
            title_prefix=title_prefix
        )

    finally:
        vis_handle.remove()
        ir_handle.remove()

    print(f"Results saved to: {save_dir}")


def get_args():
    parser = argparse.ArgumentParser("AFF response visualization")

    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="sysu", type=str)
    parser.add_argument("--arch", default="resnet50", type=str)
    parser.add_argument("--img-h", default=384, type=int)
    parser.add_argument("--img-w", default=144, type=int)

    parser.add_argument(
        "--ckpt",
        default="save_model/sysu_deen_p4_n6_lr_0.1_seed_0_best.t",
        type=str
    )
    parser.add_argument(
        "--vis-path",
        default="/root/autodl-tmp/project/LLCM-main/DEEN/Dataset/SYSU-MM01/cam1/0001/0001.jpg",
        type=str
    )
    parser.add_argument(
        "--ir-path",
        default="/root/autodl-tmp/project/LLCM-main/DEEN/Dataset/SYSU-MM01/cam3/0001/0001.jpg",
        type=str
    )
    parser.add_argument("--save-dir", default="aff_response_results", type=str)
    parser.add_argument("--hook-after", default="conv1", choices=["conv1", "relu"])
    parser.add_argument("--reduce-type", default="abs_mean", choices=["abs_mean", "l2"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset != "sysu":
        raise ValueError("This script is currently filled with SYSU defaults. If needed, change dataset/n_class/paths.")

    n_class = 395
    model = embed_net(n_class, args.dataset, arch=args.arch)
    load_checkpoint_safely(model, args.ckpt)
    model.to(device)
    model.eval()

    visualize_aff_responses_with_hook(
        model=model,
        vis_path=args.vis_path,
        ir_path=args.ir_path,
        save_dir=args.save_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        hook_after=args.hook_after,
        reduce_type=args.reduce_type,
        device=device
    )