import gdown

import torch

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

import shutil
import glob

# # Denoise
# if not os.path.exists("./experiments/pretrained_models/NAFNet-SIDD-width64.pth"):
#   gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
# Deblur
if not os.path.exists("./experiments/pretrained_models/NAFNet-REDS-width64.pth"):
  gdown.download('https://drive.google.com/uc?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X', "./experiments/pretrained_models/", quiet=False)
# # Super Resolution
# if not os.path.exists("./experiments/pretrained_models/NAFSSR-L_4x.pth"):
#   gdown.download('https://drive.google.com/uc?id=1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb', "./experiments/pretrained_models/", quiet=False)

# # Denoise
# if not os.path.exists("./demo_input/noisy-demo-0.png"):
#   gdown.download('https://drive.google.com/uc?id=1uKwZUgeGfBYLlPKllSuzgGUItlzb40hm', "demo_input/", quiet=False)

# # Deblur
# if not os.path.exists("./demo_input/blurry-reds-0.jpg"):
#   gdown.download('https://drive.google.com/uc?id=1kWjrGsAvh4gOA_gn7rB9vnnQVfRINwEn', "demo_input/", quiet=False)

# # Super Resolution
# if not os.path.exists("./demo_input/Middlebury_lr_x4_sword2_l.png"):
#   gdown.download('https://drive.google.com/uc?id=15MLvll3frPC2ICjTUnXvnLe58D_dONXc', "demo_input/", quiet=False)
# if not os.path.exists("./demo_input/Middlebury_lr_x4_sword2_r.png"):
#   gdown.download('https://drive.google.com/uc?id=1tedRtTen7LFHXsaC4ddwwg8ZkbEHPL0b', "demo_input/", quiet=False)

# deblur_upload_folder = 'D:/YOLOv7_crop_tool/yolov7/runs/detect/exp'
deblur_upload_folder = 'D:/cmd_prac/yolov7_crop_tool/runs/detect/exp'
deblur_result_folder = 'D:/cmd_prac/deblur_tool-NAFNet-/upload/output'

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1)
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('NAFNet output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)

def single_image_inference(model, img, save_path):
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])
      imwrite(sr_img, save_path)

def sr_display(LR_l, LR_r, SR_l, SR_r):
  h,w = SR_l.shape[:2]
  LR_l = cv2.resize(LR_l, (w,h), interpolation=cv2.INTER_CUBIC)
  LR_r = cv2.resize(LR_r, (w,h), interpolation=cv2.INTER_CUBIC)
  fig = plt.figure(figsize=(w//40, h//40))
  ax1 = fig.add_subplot(2, 2, 1)
  plt.title('Input image (Left)', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(2, 2, 2)
  plt.title('NAFSSR output (Left)', fontsize=16)
  ax2.axis('off')
  ax1.imshow(LR_l)
  ax2.imshow(SR_l)

  ax3 = fig.add_subplot(2, 2, 3)
  plt.title('Input image (Right)', fontsize=16)
  ax3.axis('off')
  ax4 = fig.add_subplot(2, 2, 4)
  plt.title('NAFSSR output (Right)', fontsize=16)
  ax4.axis('off')
  ax3.imshow(LR_r)
  ax4.imshow(SR_r)

  plt.subplots_adjust(wspace=0.04, hspace=0.04)

def stereo_image_inference(model, img_l, img_r, save_path):
      img = torch.cat([img_l, img_r], dim=0)
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      img_L = visuals['result'][:,:3]
      img_R = visuals['result'][:,3:]
      img_L, img_R = tensor2img([img_L, img_R])

      imwrite(img_L, save_path.format('L'))
      imwrite(img_R, save_path.format('R'))

opt_path = 'options/test/REDS/NAFNet-width64.yml'
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)

input_list = sorted(glob.glob(os.path.join(deblur_upload_folder, '*')))

for input_path in input_list:
  img_input = imread(input_path)
  inp = img2tensor(img_input)
  output_path = os.path.join(deblur_result_folder, os.path.basename(input_path))
  single_image_inference(NAFNet, inp, output_path)

output_list = sorted(glob.glob(os.path.join(deblur_result_folder, '*')))
for input_path, output_path in zip(input_list, output_list):
  img_input = imread(input_path)
  img_output = imread(output_path)
  # display(img_input, img_output)