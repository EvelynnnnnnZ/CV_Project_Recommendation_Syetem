import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from PIL import Image
import numpy as np
import pdb
import json
import base64, gzip
import numpy as np
import hashlib
import time
import cv2
import random
import PIL.Image as Image
import glob
import imageio
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
# from util_os import *
import warnings
warnings.filterwarnings("ignore")

dinov2_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_model = dinov2_model.cuda()


file_list = glob.glob('./PokemonData/**/*.jpg')
if True:
    random_index = random.randint(0,len(file_list)-1)
    input_path = file_list[random_index]
    input = cv2.imread(input_path)
    input_object_pil = Image.fromarray(np.uint8(input))
    input_object = np.array(input_object_pil)
    input_object_tensor = dinov2_preprocess(input_object_pil)
    input_object_tensor = input_object_tensor.cuda().type(torch.float32)

    random_index = random.randint(0,len(file_list)-1)
    target_path = file_list[random_index]
    target = cv2.imread(target_path)
    target_object_pil = Image.fromarray(np.uint8(target))
    target_object = np.array(target_object_pil)
    target_object_tensor = dinov2_preprocess(target_object_pil)
    target_object_tensor = target_object_tensor.cuda().type(torch.float32)



    input_feat = dinov2_model.forward_features(input_object_tensor.unsqueeze(0))
    patch_tokens_input = input_feat['x_norm_patchtokens']  # 1x256x1536
    class_token_input = input_feat['x_norm_clstoken']

    target_feat = dinov2_model.forward_features(target_object_tensor.unsqueeze(0))
    patch_tokens_target = target_feat['x_norm_patchtokens']  # 1x256x1536
    class_token_target = target_feat['x_norm_clstoken']


    input_feat = class_token_input.detach().cpu().numpy()
    target_feat = class_token_target.detach().cpu().numpy()

    cos_sim = cosine_similarity(input_feat, target_feat)
    print(input_path)
    print(target_path)
    print(cos_sim)

