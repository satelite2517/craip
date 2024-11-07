import sys
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import tqdm
import cv2
import torch
import clip

upper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(upper_dir)

from utils.clip_mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats
from utils.mp3dcat import mp3dcat

def parse_configs():
    abs_path = os.path.dirname(os.path.abspath(__file__)) 
    ckpt_path = abs_path + "/../../models/lseg/checkpoints/demo_e200.ckpt"
    parser = argparse.ArgumentParser(description="config for vlmap creation")
    parser.add_argument('--data_path', type=str, default=os.path.join(abs_path, '../data'), metavar='PATH', help="the path for dataset required for vlmap creation")
    parser.add_argument('--video_path', type=str, default=os.path.join(abs_path, '../video'), metavar='PATH', help="the path for saving generated video")
    parser.add_argument('--pretrained_path', type=str, default=ckpt_path, metavar='PATH', help="the path for pretrained checkpoint of lseg")
    parser.add_argument('--mask_version', type=int, default=1, help='mask version | 0 | 1 |, (default: 1)')
    parser.add_argument('--map_type', type=str, default='obstacle', help='type of the map to show | obstacle | rgb | landmark | openvocab | (default: obstacle)')
    parser.add_argument('--video', action='store_true', help='determine whether to make videos for the original data')
    parser.add_argument('--map', action='store_true', help='determine whether to show generated maps')
    parser.add_argument('--lang', nargs='+', default=['big flat counter', 'sofa', 'floor', 'chair', ';', 'other'], help='default open vocabulary to query VLMap')
    parser.add_argument('--clip_version', type=str, default="ViT-B/32", help='version of CLIP')
    args = parser.parse_args()

    return args

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth

def get_fast_video_writer(video_file: str, fps: int = 60, has_gpu=True):
    codec = "h264"
    if has_gpu:
        codec = "h264_nvenc"
    if (
        "google.colab" in sys.modules
        and os.path.splitext(video_file)[-1] == ".mp4"
        and os.environ.get("IMAGEIO_FFMPEG_EXE") == "/usr/bin/ffmpeg"
    ):
        # USE GPU Accelerated Hardware Encoding
        writer = imageio.get_writer(
            video_file,
            fps=fps,
            codec=codec,
            mode="I",
            bitrate="1000k",
            format="FFMPEG",
            ffmpeg_log_level="info",
            quality=10,
            output_params=["-minrate", "500k", "-maxrate", "5000k"],
        )
    else:
        # Use software encoding
        writer = imageio.get_writer(video_file, fps=fps)
    return writer

def create_video(data_dir: str, output_dir: str, fps: int = 30, has_gpu=True):
    
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    rgb_out_path = os.path.join(output_dir, "rgb.mp4")
    depth_out_path = os.path.join(output_dir, "depth.mp4")
    rgb_writer = get_fast_video_writer(rgb_out_path, fps=fps, has_gpu=has_gpu)
    depth_writer = get_fast_video_writer(depth_out_path, fps=fps, has_gpu=has_gpu)

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pbar = tqdm.tqdm(total=len(rgb_list), position=0, leave=True)
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = load_depth(depth_path)
        depth_vis = (depth / 10 * 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        rgb_writer.append_data(rgb)
        depth_writer.append_data(depth_color)
        pbar.update(1)
    rgb_writer.close()
    depth_writer.close()

def show_obstacle_map(obstacles_save_path, obstacles, xmin, xmax, ymin, ymax):
    obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap='gray')
    plt.show()
    
def show_topdown_color_map(color_top_down_save_path, xmin, xmax, ymin, ymax):
    color_top_down = load_map(color_top_down_save_path)
    color_top_down = color_top_down[xmin:xmax+1, ymin:ymax+1]
    color_top_down_pil = Image.fromarray(color_top_down)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.show()

def show_landmark_indexing_map(grid_save_path, obstacles, xmin, xmax, ymin, ymax, device, clip_version):
    
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    grid = load_map(grid_save_path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]
    
    no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
    
    lang = mp3dcat 
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps")
    plt.imshow(seg)
    plt.show()
    
def show_openvocab_landmark_indexing_map(grid_save_path, obstacles, xmin, xmax, ymin, ymax, device, lang, clip_version):

    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    grid = load_map(grid_save_path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]
    
    no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps")
    plt.imshow(seg)
    plt.show()

if __name__ == '__main__':
    args = parse_configs()
    if args.video:
        create_video(args.data_path, args.video_path)
    
    if args.map:
        map_save_dir = os.path.join(args.data_path, "vlmaps")
        color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{args.mask_version}.npy")
        grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{args.mask_version}.npy")
        weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{args.mask_version}.npy")
        obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        obstacles = load_map(obstacles_save_path)
        x_indices, y_indices = np.where(obstacles == 0)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)

        if args.map_type == 'obstacle':
            show_obstacle_map(obstacles_save_path, obstacles, xmin, xmax, ymin, ymax)
        elif args.map_type == 'rgb':
            show_topdown_color_map(color_top_down_save_path, xmin, xmax, ymin, ymax)
        elif args.map_type == 'landmark':
            show_landmark_indexing_map(grid_save_path, obstacles, xmin, xmax, ymin, ymax, device, args.clip_version)
        elif args.map_type == 'openvocab':
            show_openvocab_landmark_indexing_map(grid_save_path, obstacles, xmin, xmax, ymin, ymax, device, args.lang, args.clip_version)
        else:
            raise NotImplementedError("{} is not implemented.".format(args.map_type))


