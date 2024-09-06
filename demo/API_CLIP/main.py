import os, time, base64, requests, json, sys, datetime, argparse
from itertools import product

from PIL import Image
import cv2

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .clip_prs.utils.factory import create_model_and_transforms, get_tokenizer
from .hook import hook_prs_logger

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

def get_model(model_name = "ViT-L-14-336", layer_index = 23, device = "cuda:0"): # "ViT-L-14", "ViT-B-32"
    ## Hyperparameters
    pretrained = 'openai' # 'laion2b_s32b_b79k'

    ## Loading Model
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size
    tokenizer = get_tokenizer(model_name)

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, device, layer_index)

    return model, prs, preprocess, device, tokenizer

def gen_mask(model, prs, preprocess, device, tokenizer, image_path_or_pil_images, questions):
    ## Load image
    images = []
    image_pils = []
    for image_path_or_pil_image in image_path_or_pil_images:
        if isinstance(image_path_or_pil_image, str):
            image_pil = Image.open(image_path_or_pil_image)
        elif isinstance(image_path_or_pil_image, Image.Image):
            image_pil = image_path_or_pil_image
        else:
            raise NotImplementedError
        image = preprocess(image_pil)[np.newaxis, :, :, :]
        images.append(image)
        image_pils.append(image_pil)
    image = torch.cat(images, dim = 0).to(device)

    ## Run the image:
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image, 
                                            attn_method='head', 
                                            normalize=False)  
                         
        attentions, mlps = prs.finalize(representation)  

    ## Get the texts
    lines = questions if isinstance(questions, list) else [questions]
    print(lines[0])
    texts = tokenizer(lines).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    attention_map = attentions[:, 0, 1:, :]
    attention_map = torch.einsum('bnd,bd->bn', attention_map, class_embedding)
    HW = int(np.sqrt(attention_map.shape[1]))
    batch_size = attention_map.shape[0]
    attention_map = attention_map.view(batch_size,HW,HW)
    # print(HW)

    token_map = torch.einsum('bnd,bd->bn', mlps[:,0,:,:], class_embedding)
    token_map = token_map.view(batch_size,HW,HW)

    return attention_map[0], token_map[0]

def merge_mask(cls_mask, patch_mask, kernel_size = 3, enhance_coe = 10):
    cls_mask = normalize(cls_mask, "min")
    cls_mask = enhance(cls_mask, coe = enhance_coe)

    patch_mask = normalize(patch_mask, "max")

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size = kernel_size, padding = padding_size, padding_mode = "replicate", stride = 1, bias = False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(cls_mask.device)

    cls_mask = conv(cls_mask.unsqueeze(0))[0]

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = normalize(cls_mask + patch_mask - cls_mask * patch_mask, "min")

    return mask

def blend_mask(image, cls_mask, patch_mask, enhance_coe, kernel_size, interpolate_method_name, grayscale):
    mask = merge_mask(cls_mask, patch_mask, kernel_size = kernel_size, enhance_coe = enhance_coe)
    mask = toImg(mask.detach().cpu().unsqueeze(0))
    interpolate_method = getattr(Image, interpolate_method_name)
    mask = invtrans(mask, image, method = interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB"), grayscale).convert("RGB")

    return merged_image