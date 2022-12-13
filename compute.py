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
    parser.add_argument('--model_path', type=str, default='/home/hallasan/nvme0n1/JK/myProject/sdr2hdr/swinir_sdr2hdr/models/620444_G.pth')
    parser.add_argument('--model', type=str, default='swinIR')
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
    else:    
        modelpaths = [path for path in sorted(os.listdir(args.model_path)) if path.endswith('_G.pth') and int(path.split('_')[0]) >= 500000]
        folder, save_dir, border, window_size = args.folder_lq, f"./inference_results/{args.model}/{modelpaths[args.index].split('_')[0]}", 0, 8

    model = define_model(args, os.path.join(args.model_path, modelpaths[args.index]))
    model.eval()
    model = model.to(device)
    criterion = torch.nn.MSELoss().cuda()

    imgnames = sorted([x for x in os.listdir(args.folder_lq) if x.endswith('.png')])
    for i in range(len(imgnames)):
        # read image
        imgname = imgnames[i]
        lq_path = os.path.join(args.folder_lq, imgname)
        
        img_lq = get_input(args, lq_path, '4K')
        img_lq = img_lq / 255.
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
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True) 
            start.record()
            # Waits for everything to finish running  
            output = model(img_lq)
            output = output[..., :h_old, :w_old]
            end.record()
            torch.cuda.synchronize()
            print(f"Time elapsed: {start.elapsed_time(end)}")
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 65535.0).round().astype(np.uint16)  # float32 to uint8

        if args.save_img:
            cv2.imwrite(f'{save_dir}/{imgname}', output)
            print(f'{save_dir}/{imgname}')

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



def get_input(args, lq_path, opt=None):
    img_lq = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if opt == '2K':
        img_lq = cv2.resize(img_lq, dsize=(1920, 1080), interpolation=cv2.INTER_NEAREST)
    elif opt == '4K':
        img_lq = cv2.resize(img_lq, dsize=(3840, 2160), interpolation=cv2.INTER_NEAREST)

    return img_lq


if __name__ == '__main__':
    main()
