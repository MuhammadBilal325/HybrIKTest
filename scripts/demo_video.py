"""Video demo script with ONNX Runtime and PyTorch HybrIK backends."""
import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

try:
    import onnxruntime as ort
except ImportError:
    ort = None


det_transform = T.Compose([T.ToTensor()])

ONNX_OUTPUT_NAMES = [
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


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), "Cannot capture source"
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    videoinfo = {"fourcc": fourcc, "fps": fps, "frameSize": frame_size}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=""):
    if ext == "mp4":
        return cv2.VideoWriter_fourcc(*"mp4v"), "." + ext
    if ext == "avi":
        return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
    if ext == "mov":
        return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
    print(f"Unknow video format {ext}, will use .mp4 instead of it")
    return cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"


def try_create_writers(base_savepath, base_savepath2d, fps, frame_size):
    codec_candidates = [
        (cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"),
        (cv2.VideoWriter_fourcc(*"XVID"), ".avi"),
        (cv2.VideoWriter_fourcc(*"MJPG"), ".avi"),
    ]

    for fourcc, ext in codec_candidates:
        savepath = os.path.splitext(base_savepath)[0] + ext
        savepath2d = os.path.splitext(base_savepath2d)[0] + ext
        w1 = cv2.VideoWriter(savepath, fourcc, fps, frame_size)
        w2 = cv2.VideoWriter(savepath2d, fourcc, fps, frame_size)
        if w1.isOpened() and w2.isOpened():
            print(f"Using video codec {ext} for output: {savepath}")
            return w1, w2, savepath, savepath2d
        w1.release()
        w2.release()

    return None, None, None, None


def load_smpl_faces(model_path="./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"):
    with open(model_path, "rb") as smpl_file:
        smpl_data = pk.load(smpl_file, encoding="latin1")
    if isinstance(smpl_data, dict):
        faces = smpl_data["f"]
    else:
        faces = smpl_data.f
    return torch.from_numpy(np.asarray(faces, dtype=np.int32))


def create_onnx_session(onnx_path, gpu_id):
    if ort is None:
        raise ImportError(
            "onnxruntime is not installed. Install onnxruntime-gpu (or onnxruntime) to use ONNX backend."
        )
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": gpu_id}))
    providers.append("CPUExecutionProvider")
    print(f"ONNX Runtime providers: {providers}")
    return ort.InferenceSession(onnx_path, providers=providers)


def onnx_outputs_to_pose_output(outputs, device):
    tensor_map = {
        name: torch.from_numpy(array).to(device=device)
        for name, array in zip(ONNX_OUTPUT_NAMES, outputs)
    }
    return edict(tensor_map)


parser = argparse.ArgumentParser(description="HybrIK Demo")
parser.add_argument("--gpu", help="gpu", default=0, type=int)
parser.add_argument("--video-name", help="video name", default="examples/dance.mp4", type=str)
parser.add_argument("--out-dir", help="output folder", default="", type=str)
parser.add_argument(
    "--hybrik-backend",
    help="hybrik inference backend",
    default="onnx",
    choices=["onnx", "torch"],
    type=str,
)
parser.add_argument(
    "--hybrik-ckpt",
    help="hybrik pytorch checkpoint for torch backend",
    default="./pretrained_models/hybrik_hrnet.pth",
    type=str,
)
parser.add_argument(
    "--hybrik-onnx",
    help="hybrik onnx model for onnx backend",
    default="./pretrained_models/hybrik_hrnet.onnx",
    type=str,
)
parser.add_argument(
    "--cfg",
    help="hybrik config file",
    default="configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml",
    type=str,
)
parser.add_argument(
    "--save-pk", default=False, dest="save_pk", help="save prediction", action="store_true"
)
parser.add_argument(
    "--save-img", default=False, dest="save_img", help="save prediction", action="store_true"
)
opt = parser.parse_args()

cfg = update_config(opt.cfg)

bbox_3d_shape = getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict(
    {
        "joint_pairs_17": None,
        "joint_pairs_24": None,
        "joint_pairs_29": None,
        "bbox_3d_shape": bbox_3d_shape,
    }
)

res_keys = [
    "pred_uvd",
    "pred_xyz_17",
    "pred_xyz_29",
    "pred_xyz_24_struct",
    "pred_scores",
    "pred_camera",
    "pred_betas",
    "pred_thetas",
    "pred_phi",
    "pred_cam_root",
    "transl",
    "transl_camsys",
    "bbox",
    "height",
    "width",
    "img_path",
]
res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLCam(
    dummpy_set,
    scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR,
    sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False,
    add_dpg=False,
    loss_type=cfg.LOSS["TYPE"],
)

det_model = fasterrcnn_resnet50_fpn(pretrained=True)
det_model.cuda(opt.gpu)
det_model.eval()

if opt.hybrik_backend == "torch":
    hybrik_model = builder.build_sppe(cfg.MODEL)
    print(f"Loading PyTorch model from {opt.hybrik_ckpt}...")
    save_dict = torch.load(opt.hybrik_ckpt, map_location="cpu")
    if isinstance(save_dict, dict) and "model" in save_dict:
        model_dict = save_dict["model"]
    else:
        model_dict = save_dict
    hybrik_model.load_state_dict(model_dict)
    hybrik_model.cuda(opt.gpu)
    hybrik_model.eval()
    smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))
    onnx_session = None
else:
    if not os.path.isfile(opt.hybrik_onnx):
        raise FileNotFoundError(
            f"ONNX model not found: {opt.hybrik_onnx}. Run scripts/convert_pth_to_onnx.py first."
        )
    print(f"Loading ONNX model from {opt.hybrik_onnx}...")
    onnx_session = create_onnx_session(opt.hybrik_onnx, opt.gpu)
    hybrik_model = None
    smpl_faces = load_smpl_faces()

print("### Extract Image...")
video_basename = os.path.basename(opt.video_name).split(".")[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, "raw_images")):
    os.makedirs(os.path.join(opt.out_dir, "raw_images"))
if not os.path.exists(os.path.join(opt.out_dir, "res_images")) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, "res_images"))
if not os.path.exists(os.path.join(opt.out_dir, "res_2d_images")) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, "res_2d_images"))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split(".")[0]

savepath = f"./{opt.out_dir}/res_{video_basename}.mp4"
savepath2d = f"./{opt.out_dir}/res_2d_{video_basename}.mp4"
info["savepath"] = savepath
info["savepath2d"] = savepath2d

write_stream, write2d_stream, final_savepath, final_savepath2d = try_create_writers(
    info["savepath"], info["savepath2d"], info["fps"], info["frameSize"]
)
if write_stream is None or write2d_stream is None:
    print("No compatible video encoder found in OpenCV. Falling back to image output only.")
    if not opt.save_img:
        print("Enabling --save-img automatically so results are still written.")
        opt.save_img = True
else:
    info["savepath"] = final_savepath
    info["savepath2d"] = final_savepath2d

os.system(f"ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png")

files = os.listdir(f"{opt.out_dir}/raw_images")
files.sort()

img_path_list = []
for file_name in tqdm(files):
    if not os.path.isdir(file_name) and file_name[-4:] in [".jpg", ".png"]:
        img_path = os.path.join(opt.out_dir, "raw_images", file_name)
        img_path_list.append(img_path)

prev_box = None
device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")

print("### Run Model...")
idx = 0
for img_path in tqdm(img_path_list):
    with torch.no_grad():
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)
            if tight_bbox is None:
                continue
        else:
            tight_bbox = get_max_iou_box(det_output, prev_box)

        prev_box = tight_bbox

        pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]

        if opt.hybrik_backend == "torch":
            pose_output = hybrik_model(
                pose_input,
                flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            )
        else:
            ort_inputs = {
                "img": np.ascontiguousarray(pose_input.detach().cpu().numpy().astype(np.float32)),
                "bboxes": np.asarray(bbox, dtype=np.float32).reshape(1, 4),
                "img_center": np.asarray(img_center, dtype=np.float32).reshape(1, 2),
            }
            ort_outputs = onnx_session.run(None, ort_inputs)
            pose_output = onnx_outputs_to_pose_output(ort_outputs, device)

        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()

        image = input_image.copy()
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = transl.clone()
        transl_camsys = transl_camsys * 256 / bbox_xywh[2]

        focal = focal / 256 * bbox_xywh[2]

        vertices = pose_output.pred_vertices.detach()

        verts_batch = vertices
        transl_batch = transl

        color_batch = render_mesh(
            vertices=verts_batch,
            faces=smpl_faces,
            translation=transl_batch,
            focal_length=focal,
            height=image.shape[0],
            width=image.shape[1],
        )

        valid_mask_batch = color_batch[:, :, :, [-1]] > 0
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = image
        alpha = 0.9
        image_vis = (
            alpha * color[:, :, :3] * valid_mask
            + (1 - alpha) * input_img * valid_mask
            + (1 - valid_mask) * input_img
        )

        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        if opt.save_img:
            idx += 1
            res_path = os.path.join(opt.out_dir, "res_images", f"image-{idx:06d}.jpg")
            cv2.imwrite(res_path, image_vis)
        if write_stream is not None:
            write_stream.write(image_vis)

        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        image = input_image.copy()
        bbox_img = vis_2d(image, tight_bbox, pts)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        if write2d_stream is not None:
            write2d_stream.write(bbox_img)

        if opt.save_img:
            res_path = os.path.join(opt.out_dir, "res_2d_images", f"image-{idx:06d}.jpg")
            cv2.imwrite(res_path, bbox_img)

        if opt.save_pk:
            assert pose_input.shape[0] == 1, "Only support single batch inference for now"

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(-1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            res_db["pred_xyz_17"].append(pred_xyz_jts_17)
            res_db["pred_uvd"].append(pred_uvd_jts)
            res_db["pred_xyz_29"].append(pred_xyz_jts_29)
            res_db["pred_xyz_24_struct"].append(pred_xyz_jts_24_struct)
            res_db["pred_scores"].append(pred_scores)
            res_db["pred_camera"].append(pred_camera)
            res_db["pred_betas"].append(pred_betas)
            res_db["pred_thetas"].append(pred_theta)
            res_db["pred_phi"].append(pred_phi)
            res_db["pred_cam_root"].append(pred_cam_root)
            res_db["transl"].append(transl[0].cpu().data.numpy())
            res_db["transl_camsys"].append(transl_camsys[0].cpu().data.numpy())
            res_db["bbox"].append(np.array(bbox))
            res_db["height"].append(img_size[0])
            res_db["width"].append(img_size[1])
            res_db["img_path"].append(img_path)

if opt.save_pk:
    n_frames = len(res_db["img_path"])
    for key in res_db.keys():
        print(key)
        res_db[key] = np.stack(res_db[key])
        assert res_db[key].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, "res.pk"), "wb") as fid:
        pk.dump(res_db, fid)

if write_stream is not None:
    write_stream.release()
if write2d_stream is not None:
    write2d_stream.release()
