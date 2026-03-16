"""Convert HybrIK checkpoints (.pth) to ONNX for inference deployment.

This script keeps the original .pth checkpoints unchanged and writes .onnx
files next to them (or in a custom output directory).
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import warnings

import torch

from hybrik.models import builder
from hybrik.utils.config import update_config


DEFAULT_PRETRAINED_EXPORTS = {
    "hybrik_hrnet.pth": "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml",
    "hybrikx_hrnet.pth": "configs/smplx/256x192_hrnet_smplx_kid.yaml",
    "hybrikx_rle_hrnet.pth": "configs/smplx/256x192_hrnet_rle_smplx_kid.yaml",
}

OUTPUT_NAMES = [
    "pred_uvd_jts",
    "pred_xyz_jts_17",
    "pred_xyz_jts_29",
    "pred_xyz_jts_24_struct",
    "maxvals",
    "pred_camera",
    "pred_shape",
    "pred_theta_mats",
    "pred_phi",
    "cam_root",
    "transl",
    "pred_vertices",
]

SMPLX_OUTPUT_NAMES = [
    "pred_uvd_jts",
    "pred_xyz_hybrik",
    "pred_xyz_hybrik_struct",
    "pred_xyz_full",
    "maxvals",
    "pred_camera",
    "pred_beta",
    "pred_theta_mat",
    "pred_phi",
    "cam_root",
    "transl",
    "pred_vertices",
]


class HybrikOnnxExportWrapper(torch.nn.Module):
    """Expose only tensor outputs that are required by demo inference."""

    def __init__(self, model: torch.nn.Module, output_fields: List[str]):
        super().__init__()
        self.model = model
        self.output_fields = output_fields

    def forward(self, img: torch.Tensor, bboxes: torch.Tensor, img_center: torch.Tensor):
        # Export through the training-mode forward path to bypass update_scale,
        # which uses torch.inverse and breaks on older ONNX symbolics.
        was_training = self.model.training
        self.model.train()
        out = self.model(img, flip_test=False, bboxes=bboxes, img_center=img_center)
        if not was_training:
            self.model.eval()
        return tuple(getattr(out, field) for field in self.output_fields)


def get_output_names(cfg) -> List[str]:
    model_type = str(getattr(cfg.MODEL, "TYPE", ""))
    if "SMPLX" in model_type.upper():
        return SMPLX_OUTPUT_NAMES
    return OUTPUT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HybrIK .pth checkpoints to ONNX")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a single .pth checkpoint")
    parser.add_argument("--cfg", type=str, default="", help="Config .yaml for the single checkpoint")
    parser.add_argument("--pretrained-dir", type=str, default="pretrained_models", help="Directory containing .pth files")
    parser.add_argument("--out-dir", type=str, default="", help="Directory to save .onnx files (default: same as checkpoint)")
    parser.add_argument("--all-pretrained", action="store_true", help="Convert known checkpoints from pretrained dir")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size")
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Preferred ONNX opset version (auto-fallback to lower supported opsets on failure)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ONNX files")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch axis. Default is fixed batch export for stable tracing.",
    )
    parser.add_argument(
        "--show-tracer-warnings",
        action="store_true",
        help="Show torch tracer warnings during export (hidden by default).",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    save_dict = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = save_dict["model"] if isinstance(save_dict, dict) and "model" in save_dict else save_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def get_input_hw(cfg) -> Tuple[int, int]:
    image_size = cfg.MODEL.IMAGE_SIZE
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    size = int(image_size)
    return size, size


def normalize_cfg_for_export(cfg):
    """Backfill known config keys needed by model constructors.

    Some released SMPL-X configs place flags such as USE_KID under DATASET,
    while model constructors read them from MODEL.EXTRA.
    """
    model_type = str(getattr(cfg.MODEL, "TYPE", ""))
    extra = getattr(cfg.MODEL, "EXTRA", None)
    dataset = getattr(cfg, "DATASET", None)

    if extra is None or dataset is None:
        return cfg

    if "Kid" in model_type and "USE_KID" not in extra:
        extra["USE_KID"] = bool(getattr(dataset, "USE_KID", True))

    if "HAND_REL" not in extra and hasattr(dataset, "HAND_REL"):
        extra["HAND_REL"] = bool(getattr(dataset, "HAND_REL"))

    # Export uses checkpoint weights loaded after model construction.
    # Avoid constructor-time failures when optional HRNet pretrain files are absent.
    if hasattr(cfg.MODEL, "HR_PRETRAINED"):
        cfg.MODEL.HR_PRETRAINED = ""

    return cfg


def export_one_checkpoint(
    checkpoint_path: Path,
    cfg_path: Path,
    out_dir: Path,
    batch_size: int,
    opset: int,
    overwrite: bool,
    dynamic_batch: bool,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = update_config(str(cfg_path))
    cfg = normalize_cfg_for_export(cfg)
    model = builder.build_sppe(cfg.MODEL)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    output_names = get_output_names(cfg)
    wrapper = HybrikOnnxExportWrapper(model, output_names).eval()

    h, w = get_input_hw(cfg)
    dummy_img = torch.randn(batch_size, 3, h, w, dtype=torch.float32)
    dummy_bbox = torch.tensor([[0.0, 0.0, float(w), float(h)]], dtype=torch.float32).repeat(batch_size, 1)
    dummy_center = torch.tensor([[float(w) * 0.5, float(h) * 0.5]], dtype=torch.float32).repeat(batch_size, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{checkpoint_path.stem}.onnx"
    if onnx_path.exists() and not overwrite:
        print(f"[SKIP] {onnx_path} already exists. Use --overwrite to regenerate.")
        return

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "img": {0: "batch"},
            "bboxes": {0: "batch"},
            "img_center": {0: "batch"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    capped = min(opset, 16)
    preferred_opsets = list(range(capped, 10, -1))
    tried = []
    for current_opset in preferred_opsets:
        if current_opset in tried:
            continue
        tried.append(current_opset)
        try:
            print(f"[INFO] Exporting {checkpoint_path.name} -> {onnx_path} (opset={current_opset})")
            torch.onnx.export(
                wrapper,
                (dummy_img, dummy_bbox, dummy_center),
                str(onnx_path),
                export_params=True,
                # Training-mode export is used to bypass update_scale/inverse;
                # keep constant folding off to avoid parameter mutation issues.
                do_constant_folding=False,
                opset_version=current_opset,
                input_names=["img", "bboxes", "img_center"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                training=torch.onnx.TrainingMode.TRAINING,
            )
            print(f"[OK] Wrote ONNX: {onnx_path} (opset={current_opset})")
            return
        except ValueError as exc:
            msg = str(exc)
            if "Unsupported ONNX opset version" in msg:
                print(f"[WARN] {msg}. Trying lower opset...")
                continue
            raise
        except RuntimeError as exc:
            msg = str(exc)
            if "Occurred when translating split" in msg or "symbolic_opset" in msg:
                print(f"[WARN] Export failed at opset={current_opset}: {msg.splitlines()[-1]}. Trying lower opset...")
                continue
            raise
        except TypeError as exc:
            msg = str(exc)
            if "Occurred when translating split" in msg or "NoneType' object is not subscriptable" in msg:
                print(f"[WARN] Export failed at opset={current_opset}: split symbolic issue. Trying lower opset...")
                continue
            raise

    raise RuntimeError(
        f"Failed to export {checkpoint_path.name}. None of these opsets worked: {tried}"
    )


def main():
    args = parse_args()
    if not args.show_tracer_warnings:
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    root = Path(__file__).resolve().parent.parent

    tasks: List[Tuple[Path, Path]] = []

    if args.checkpoint:
        if not args.cfg:
            raise ValueError("--cfg is required when --checkpoint is provided")
        tasks.append((Path(args.checkpoint), Path(args.cfg)))
    else:
        pretrained_dir = (root / args.pretrained_dir).resolve() if not Path(args.pretrained_dir).is_absolute() else Path(args.pretrained_dir)
        if not pretrained_dir.exists():
            raise FileNotFoundError(f"Pretrained directory not found: {pretrained_dir}")

        if args.all_pretrained:
            for ckpt_name, cfg_rel in DEFAULT_PRETRAINED_EXPORTS.items():
                ckpt_path = pretrained_dir / ckpt_name
                if ckpt_path.exists():
                    tasks.append((ckpt_path, (root / cfg_rel).resolve()))
                else:
                    print(f"[WARN] Missing checkpoint, skipping: {ckpt_path}")
        else:
            raise ValueError("Provide --checkpoint/--cfg, or use --all-pretrained")

    failed = []
    for ckpt_path, cfg_path in tasks:
        out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
        try:
            export_one_checkpoint(
                checkpoint_path=ckpt_path.resolve(),
                cfg_path=cfg_path.resolve(),
                out_dir=out_dir.resolve(),
                batch_size=args.batch_size,
                opset=args.opset,
                overwrite=args.overwrite,
                dynamic_batch=args.dynamic_batch,
            )
        except Exception as exc:
            failed.append((ckpt_path.name, str(exc)))
            print(f"[ERROR] Failed exporting {ckpt_path.name}: {exc}")

    if failed:
        print("[SUMMARY] Some checkpoints failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    else:
        print("[SUMMARY] All requested checkpoints exported successfully.")


if __name__ == "__main__":
    main()
