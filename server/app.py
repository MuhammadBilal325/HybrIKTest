import os
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from easydict import EasyDict as edict
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_one_box

app = Flask(__name__)

# --- Configuration ---
GPU_ID = 0
DEVICE = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
CFG_FILE = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = 'pretrained_models/hybrik_hrnet.pth'

# --- Initialization ---
print("Loading configuration...")
cfg = update_config(CFG_FILE)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_transform = T.Compose([T.ToTensor()])

print("Initializing models...")
# Detection model
det_model = fasterrcnn_resnet50_fpn(pretrained=True)
det_model.to(DEVICE)
det_model.eval()

# HybrIK model
hybrik_model = builder.build_sppe(cfg.MODEL)
print(f'Loading HybrIK model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if isinstance(save_dict, dict):
    model_dict = save_dict.get('model', save_dict)
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

hybrik_model.to(DEVICE)
hybrik_model.eval()
print("Models initialized successfully.")

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided in request'}), 400
    
    file = request.files['image']
    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            # 1. Human Detection
            det_input = det_transform(img_rgb).to(DEVICE)
            det_output = det_model([det_input])[0]
            tight_bbox = get_one_box(det_output)  # [x1, y1, x2, y2]
            
            if tight_bbox is None:
                return jsonify({'error': 'No person detected in the image'}), 404

            # 2. HybrIK Preprocessing
            pose_input, bbox, img_center = transformation.test_transform(img_rgb, tight_bbox)
            pose_input = pose_input.to(DEVICE)[None, :, :, :]
            
            # 3. HybrIK Inference
            # bboxes and img_center need to be tensors on the same device
            bboxes_ts = torch.from_numpy(np.array(bbox)).to(DEVICE).unsqueeze(0).float()
            img_center_ts = torch.from_numpy(img_center).to(DEVICE).unsqueeze(0).float()
            
            pose_output = hybrik_model(
                pose_input, 
                flip_test=True,
                bboxes=bboxes_ts,
                img_center=img_center_ts
            )
            
            # 4. Extract and Format Results
            # Converting tensors to lists for JSON serialization
            res = {
                'pred_phi': pose_output.pred_phi.cpu().numpy().tolist(),
                'pred_shape': pose_output.pred_shape.cpu().numpy().tolist(),
                'pred_theta_mats': pose_output.pred_theta_mats.cpu().numpy().tolist(),
                'transl': pose_output.transl.cpu().numpy().tolist(),
                'pred_vertices': pose_output.pred_vertices.cpu().numpy().tolist(),
                'pred_uvd_jts': pose_output.pred_uvd_jts.cpu().numpy().tolist(),
                'pred_xyz_jts_29': pose_output.pred_xyz_jts_29.cpu().numpy().tolist(),
                'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                'tight_bbox': tight_bbox.tolist() if isinstance(tight_bbox, np.ndarray) else tight_bbox
            }
            
            return jsonify(res)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You might want to change the port if 5000 is occupied
    app.run(host='0.0.0.0', port=5000, debug=False)
