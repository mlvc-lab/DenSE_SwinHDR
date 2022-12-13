import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

random.seed(2021)

class DatasetSDR2HDR(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(DatasetSDR2HDR, self).__init__()
        self.opt = opt
        self.data_type = "img"#self.opt['data_type']
        self.paths_L, self.paths_H = None, None
        self.sizes_L, self.sizes_H = None, None
        self.L_env, self.H_env = None, None  # environment for lmdb

        self.sizes_H, self.paths_H = util.get_image_paths(self.data_type, opt['dataroot_H'])
        self.sizes_L, self.paths_L = util.get_image_paths(self.data_type, opt['dataroot_L'])
        assert self.paths_H, 'Error: H path is empty.'

        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(
                self.paths_H
            ), 'L/H mismatch - {}, {}.'.format(
                len(self.paths_L), len(self.paths_H))
        
        '''
        for random crop
        '''
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        

    def __getitem__(self, index):
        H_path, L_path = None, None
        scale = self.opt['scale']
        H_size = self.opt['H_size']

        # get GT image
        H_path = self.paths_H[index]
        img_H = util.read_img(self.H_env, H_path)
        
        # get LQ image
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = util.read_img(self.L_env, L_path)
            
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape
            H_gt, W_gt, C = img_H.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(L_path))
            L_size = H_size // scale

            # randomly crop
            
            if H_size is not None:
                rnd_h = random.randint(0, max(0, H - L_size))
                rnd_w = random.randint(0, max(0, W - L_size))
                img_L = img_L[rnd_h:rnd_h + L_size, rnd_w:rnd_w + L_size, :]
                rnd_h_H, rnd_w_H = int(rnd_h * scale), int(rnd_w * scale)
                img_H = img_H[rnd_h_H:rnd_h_H + H_size, rnd_w_H:rnd_w_H + H_size, :]
            
            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)


        # BGR to RGB, HWC to CHW, numpy to tensor
        
        
        img_H = img_H[:, :, [2, 1, 0]]
        img_L = img_L[:, :, [2, 1, 0]]

        img_H = torch.from_numpy(np.ascontiguousarray(np.transpose(img_H, (2, 0, 1)))).float()
        img_L = torch.from_numpy(np.ascontiguousarray(np.transpose(img_L, (2, 0, 1)))).float()
      
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)