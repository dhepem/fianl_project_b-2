import os
import cv2
import torch
from image_process import load_img, save_img
import torch.nn.functional as F
from skimage import img_as_ubyte
import torch.nn as nn
from model import Restormer

img_multiple_of = 8


def motion_deblurring(img):
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4, 'heads': [
        1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
    model = Restormer(**parameters)
    weights = os.path.join('weigths', 'motion_deblurring.pth')
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'], strict=False)
    device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        input_ = torch.from_numpy(img).float().div(
            255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * \
            img_multiple_of, ((width + img_multiple_of) //
                              img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        return restored


def denoising(img):
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4, 'heads': [
        1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
    parameters['LayerNorm_type'] = 'BiasFree'
    weights = os.path.join('weigths', 'real_denoising.pth')
    model = Restormer(**parameters)
    device = torch.device('cpu')
    checkpoint = torch.load(weights)
    model.eval()
    model.load_state_dict(checkpoint['params'], strict=False)

    with torch.no_grad():
        input_ = torch.from_numpy(img).float().div(
            255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * \
            img_multiple_of, ((width + img_multiple_of) //
                              img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        return restored


def processAI(img):

    res = motion_deblurring(img)
    res = denoising(res)
    return res
