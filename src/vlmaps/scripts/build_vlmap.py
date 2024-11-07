import os
import sys
import math

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import clip
import argparse
import yaml

upper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
upper_upper_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(upper_dir)
sys.path.append(upper_upper_dir)

from utils.clip_mapping_utils import load_depth, load_pose, load_semantic, load_obj2cls_dict, save_map, cvt_obj_id_2_cls_id, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point
from models.lseg.additional_utils.models import resize_image, pad_image, crop_image
from models.lseg.modules.models.lseg_net import LSegEncNet


def parse_configs():
    abs_path = os.path.dirname(os.path.abspath(__file__)) 
    ckpt_path = abs_path + "/../../models/lseg/checkpoints/demo_e200.ckpt"
    cfg_path = abs_path + "/../../limbo/config/config.yaml"
    parser = argparse.ArgumentParser(description="config for vlmap creation")
    parser.add_argument('--data_path', type=str, default=os.path.join(abs_path, '../data'), metavar='PATH', help="the path for dataset required for vlmap creation")
    parser.add_argument('--pretrained_path', type=str, default=ckpt_path, metavar='PATH', help="the path for pretrained checkpoint of lseg")
    parser.add_argument('--cfg_path', type=str, default=cfg_path, metavar='PATH', help="the path for pretrained checkpoint of lseg")
    parser.add_argument('--mask_version', type=int, default=1, help='mask version | 0 | 1 |, (default: 1)')
    parser.add_argument('--camera_height', type=float, default=1.5, help='height of camera attached to the robot')
    parser.add_argument('--init_rot', nargs='+', action='append', default=[[0, -1, 0],[1, 0, 0],[0, 0, 1]], help='initial rotation matrix of gazebo world')
    parser.add_argument('--rot_ro_cam', nargs='+', action='append', default=[[0, 0, 1],[-1, 0, 0],[0, -1, 0]], help='camera rotation matrix')
    
    args = parser.parse_args()

    return args

def create_lseg_map_batch(pretrained_path, img_save_dir, camera_height, init_rot, rot_ro_cam, cs=0.05, gs=1000, depth_sample_rate=100, 
                          crop_size = 480, base_size = 520, lang = "door,chair,ground,ceiling,other",
                          clip_version = "ViT-B/32", mask_version=1):

    labels = lang.split(",")

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels)
    lang_token = lang_token.to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    model = LSegEncNet(lang, arch_option=0,
                        block_depth=0,
                        activation='lrelu',
                        crop_size=crop_size)
    model_state_dict = model.state_dict()
    
    if torch.cuda.is_available():
        pretrained_state_dict = torch.load(pretrained_path)
    else:
        pretrained_state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))

    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.to(device)

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    print(f"loading scene {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    pose_dir = os.path.join(img_save_dir, "pose")
    
    use_gt = False
    if os.path.isdir(os.path.join(img_save_dir, "semantic")) and os.path.isfile(os.path.join(img_save_dir, "obj2cls_dict.txt")):
        use_gt = True
        semantic_dir = os.path.join(img_save_dir, "semantic")
        obj2cls_path = os.path.join(img_save_dir, "obj2cls_dict.txt")

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    
    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]
    
    if use_gt:
        semantic_list = sorted(os.listdir(semantic_dir), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]

    map_save_dir = os.path.join(img_save_dir, "vlmaps")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    if use_gt:
        obj2cls = load_obj2cls_dict(obj2cls_path)

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)
    
    if use_gt:
        gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
        gt = np.zeros((gs, gs), dtype=np.int32)
            
    if use_gt:
        data_iter = zip(rgb_list, depth_list, semantic_list, pose_list)
    else:
        data_iter = zip(rgb_list, depth_list, pose_list)
    
    init_tf = np.eye(4)
    init_tf[:3, :3] = init_rot @ rot_ro_cam
    init_tf_inv = np.linalg.inv(init_tf)
    
    pbar = tqdm(total=len(rgb_list))
    # load all images and depths and poses
    for data_sample in data_iter:
        if use_gt:
            rgb_path, depth_path, semantic_path, pose_path = data_sample
        else:
            rgb_path, depth_path, pose_path = data_sample

        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # read pose
        pos, rot = load_pose(pose_path)
        rot = rot @ rot_ro_cam
        pos[2] += camera_height

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf = init_tf_inv @ pose

        # read depth
        depth = load_depth(depth_path)

        if use_gt:
            # read semantic
            semantic = load_semantic(semantic_path)
            semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size, norm_mean, norm_std)

        # transform all points to the global frame
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        # project all point cloud onto the ground
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(gs, cs, p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            if use_gt:
                semantic_v = semantic[rgb_py, rgb_px]
                if semantic_v == 40:
                    semantic_v = -1

            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
                if use_gt:
                    gt[y, x] = semantic_v

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > camera_height:
                continue
            obstacles[y, x] = 0
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    if use_gt:
        save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    vis_image = image.copy()
    if torch.cuda.is_available():
        image = transform(image).unsqueeze(0).cuda()
    else:
        image = transform(image).unsqueeze(0)
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height


    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            if torch.cuda.is_available():
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            else:
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        logits_outputs = logits_outputs[:,:,:height,:width]
    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]

    return outputs


if __name__ == '__main__':
    configs = parse_configs()

    with open(configs.cfg_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

    map_resolution = yaml_data['map_resolution']
    img_save_dir = configs.data_path
    pretrained_path = configs.pretrained_path

    camera_height = configs.camera_height
    init_rot = np.array(configs.init_rot)
    rot_ro_cam = np.array(configs.rot_ro_cam)

    create_lseg_map_batch(pretrained_path, img_save_dir, camera_height, init_rot, rot_ro_cam, cs=map_resolution, gs=250, depth_sample_rate=100, mask_version=configs.mask_version)
