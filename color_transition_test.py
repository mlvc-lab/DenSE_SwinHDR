import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from math import log10

from models.network_swinir import DenseSwinIR as net_Dense
from models.network_swinir import SwinIR as net
from models.network_swinir import SwinIR_D as net_D
from models.network_swinir import DenseSwinIR_withSE as net_Dense_SE
from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_patch_size', type=int, default=120, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str, default='/home/hallasan/nvme0n1/JK/myProject/sdr2hdr_Dense_se/swinir_sdr2hdr_Dense_se/models/1029838_G.pth') # 813276 Dense, 620444 IR
    parser.add_argument('--model', type=str, default='swinIR_Dense_withSE')
    parser.add_argument('--folder_lq', type=str, default='/home/hallasan/nvme0n1/HDRTV_dataset/test/test_sdr_1K_gpu', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='/home/hallasan/nvme0n1/HDRTV_dataset/test/test_hdr_1K_gpu', help='input ground-truth test image folder')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    
    if args.model_path.endswith(".pth"):
        modelpaths = [args.model_path]
        args.index = 0
        folder, save_dir, border, window_size = args.folder_lq, f"./inference_results/{args.model}/{modelpaths[args.index].split('/')[-1].split('_')[0]}", 0, 8
        os.makedirs(save_dir, exist_ok=True)
    
    model = define_model(args, os.path.join(args.model_path, modelpaths[args.index]))
    model.eval()
    model = model.to(device)
    imgPath = "/home/hallasan/nvme0n1/JK/color.png"
    img_lq = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old, :w_old]
    
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 65535.0).round().astype(np.uint16)  # float32 to uint8    
    cv2.imwrite(f'{save_dir}/transition_result.png', output)

def define_model(args, modelpath):
    
    # SwinIR
    if args.model == 'swinIR':
        model = net(upscale=1, in_chans=3, img_size=120, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler=None, resi_connection='1conv')
    elif args.model == 'swinIR_D':
    # SwinIR_D
        model = net_D(upscale=1, in_chans=3, img_size=120, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler=None, resi_connection='1conv')
    elif args.model == 'swinIR_Dense':
    # SwinIR_Dense
        model = net_Dense(growth_rate=30,
                    upscale=1,
                    in_chans=3,
                    img_size=120,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler=None,
                    resi_connection='1conv')
    elif args.model == 'swinIR_Dense_withSE':
        model = net_Dense_SE(growth_rate=30,
                    upscale=1,
                    in_chans=3,
                    img_size=120,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler=None,
                    resi_connection='1conv')
                    
    model.load_state_dict(torch.load(modelpath), strict=True)
    return model

if __name__ == '__main__':
    main()
