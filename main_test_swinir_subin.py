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
    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str, default='/root/subin/DenSE_Swin_SDR2HDR/sdr2hdr_Dense_se/swinir_sdr2hdr_Dense_se_test/models/10_G.pth')
    parser.add_argument('--model', type=str, default='swinIR_Dense_withSE')
    parser.add_argument('--folder_lq', type=str, default='/root/subin/HDRTVNet/dataset/test_set/test_sdr', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='/root/subin/HDRTVNet/dataset/test_set/test_hdr', help='input ground-truth test image folder')
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
        f = open(f"{save_dir}/{modelpaths[args.index].split('/')[-1].split('_')[0]}.txt", 'w')
        print(f"{save_dir}/{modelpaths[args.index].split('/')[-1].split('_')[0]}.txt")
    else:    
        modelpaths = [path for path in sorted(os.listdir(args.model_path)) if path.endswith('_G.pth') and int(path.split('_')[0]) >= 500000]
        folder, save_dir, border, window_size = args.folder_lq, f"./inference_results/{args.model}/{modelpaths[args.index].split('_')[0]}", 0, 8
        os.makedirs(save_dir, exist_ok=True)
        f = open(f"{save_dir}/{modelpaths[args.index].split('_')[0]}.txt", 'w')
        print(f"{save_dir}/{modelpaths[args.index].split('/')[-1].split('_')[0]}.txt")

    model = define_model(args, os.path.join(args.model_path, modelpaths[args.index]))
    model.eval()
    model = model.to(device)
    criterion = torch.nn.MSELoss().cuda()
    
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    imgnames = sorted([x for x in os.listdir(args.folder_lq) if x.endswith('.png')])
    for i in range(len(imgnames)):
        # read image
        imgname = imgnames[i]
        lq_path = os.path.join(args.folder_lq, imgname)
        gt_path = os.path.join(args.folder_gt, imgname)
        img_lq, img_gt = get_image_pair(args, lq_path, gt_path)  # image to HWC-BGR, float32
        img_lq, img_gt_scaled = img_lq / 255., img_gt / 65535.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        img_gt_t = np.transpose(img_gt_scaled if img_gt_scaled.shape[2] == 1 else img_gt_scaled[:, :, [2, 1, 0]], (2, 0, 1))
        img_gt_t = torch.from_numpy(img_gt_t).float().unsqueeze(0).to(device)

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

        mse = criterion(output, img_gt_t)
        psnr_t = 10 * log10(1 / mse.item())
        test_results['psnr'].append(psnr_t)
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 65535.0).round().astype(np.uint16)  # float32 to uint8

        ssim = util.calculate_ssim(output, img_gt, 16)
        test_results['ssim'].append(ssim)

        print('Testing {:d} {:20s} -  PSNR TORCH: {:.2f} dB SSIM {:.4f}'.
                format(i, imgname, psnr_t, ssim))
        f.write('Testing {:d} {:20s} -  PSNR TORCH: {:.2f} dB SSIM {:.4f}\n'.
                format(i, imgname, psnr_t, ssim))
        # save image
        if args.save_img:
            cv2.imwrite(f'{save_dir}/{imgname}', output)

        # evaluate psnr/ssim/psnr_b
        '''
        if img_gt is not None:
            img_gt = np.squeeze(img_gt)
            output = output.astype(np.float32) / 65535.0
            psnr = util.calculate_psnr(output, img_gt)
            test_results['psnr'].append(psnr)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB PSNR TORCH: {:.2f}'.
                  format(i, imgname, psnr, psnr_t))
        else:
            print('Testing {:d} {:20s}'.format(i, imgname))
        '''

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR: {:.2f} dB'.format(save_dir, ave_psnr))
        print('\n{} \n-- Average SSIM: {:.4f}'.format(save_dir, ave_ssim))
        
        f.write('\n{} \n-- Average PSNR: {:.2f} dB'.format(save_dir, ave_psnr))
        f.write('\n{} \n-- Average SSIM: {:.4f}'.format(save_dir, ave_ssim))
        f.close()


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
                    img_size=64,
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



def get_image_pair(args, lq_path, gt_path, opt=None):
    img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_lq = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if opt == '2K':
        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_lq = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    elif opt == '4K':
        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_lq = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    return img_lq, img_gt


if __name__ == '__main__':
    main()
