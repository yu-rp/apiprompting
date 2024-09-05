## Imports
import os, time, argparse, base64, requests, os, json, sys, datetime
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# import cv2
from PIL import Image

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as T

from functions import getmask, get_model
from hook import hook_logger
from DatasetManager.dataloader import get_data

def readImg(p):
    return Image.open(p)

def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method = Image.BICUBIC):
    return mask.resize(image.size, method)

def merge(mask, image, grap_scale = 200):
    gray = np.ones((image.size[1], image.size[0], 3))*grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None]  + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image

def normalize(mat, method = "max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0,1)
    return mat

def revise_mask(patch_mask, kernel_size = 3, enhance_coe = 10):

    patch_mask = normalize(patch_mask, "min")
    patch_mask = enhance(patch_mask, coe = enhance_coe)

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size = kernel_size, padding = padding_size, padding_mode = "replicate", stride = 1, bias = False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(patch_mask.device)

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = patch_mask

    return mask

def blend_mask(image_path_or_pil_image, mask, key, enhance_coe, kernel_size, interpolate_method, folder):
    mask = revise_mask(mask.float(), kernel_size = kernel_size, enhance_coe = enhance_coe)
    mask = mask.detach().cpu()
    mask = toImg(mask.reshape(1,24,24))

    if isinstance(image_path_or_pil_image, str):
        image = readImg(image_path_or_pil_image)
    elif isinstance(image_path_or_pil_image, Image.Image):
        image = image_path_or_pil_image
    else:
        raise NotImplementedError

    mask = invtrans(mask, image, method = interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB")).convert("RGB")

    file_name = os.path.join(folder, f"{key}.jpg")
    merged_image.save(file_name)
    print(file_name)
    return merged_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mmvet")
    parser.add_argument("--layer_index", type=int, default=20)
    parser.add_argument("--model", type=str, default="13b")
    parser.add_argument("--range", type=int, nargs = 2, default=(0,0))
    parser.add_argument("--output_folder", type=str, default = "../../experiments")
    parser.add_argument("--interpolate_method_name", type=str, default = "LANCZOS")
    parser.add_argument("--enhance_coe", type=int, default = 5)
    parser.add_argument("--kernel_size", type=int, default = 3)
    parser.add_argument("--grayscale", type=int, default = 0)

    args = parser.parse_args()

    exp_folder = f"{args.output_folder}/APILLaVA_{args.dataset}_{args.model}_{args.layer_index}"

    tokenizer, model, image_processor, context_len, model_name = get_model("llava-v1.5-7b" if args.model == "7b" else "llava-v1.5-13b")
    hl = hook_logger(model, model.device, layer_index = args.layer_index)
    
    dataset = get_data(args.dataset)

    enhance_coe = args.enhance_coe
    kernel_size = args.kernel_size
    grayscale = args.grayscale
    interpolate_method_name = args.interpolate_method_name
    interpolate_method = getattr(Image, interpolate_method_name)
    mask_image_folder = os.path.join(exp_folder, f"{enhance_coe}_{kernel_size}_{interpolate_method_name}")
    os.makedirs(mask_image_folder, exist_ok = True)

    for qindex in range(len(dataset)):
        if args.range[1] > 0:
            if qindex < args.range[0] or qindex >= args.range[1]:
                continue

        key, image_path_or_pil_image, question = dataset.get_item(qindex)
        # print(key, image_path_or_pil_image, question)

        with torch.no_grad():
            mask_args = type('Args', (), {
                    "hl":      hl,
                    "model_name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor": image_processor,
                    "context_len": context_len,
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_path_or_pil_image,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 20,
                })()

            mask, output = getmask(mask_args)

            merged_image = blend_mask(image_path_or_pil_image, mask, key, enhance_coe, kernel_size, interpolate_method, mask_image_folder)
