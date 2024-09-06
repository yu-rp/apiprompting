import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.transformers.generation.stopping_criteria import MaxNewTokensCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_preanswer(model, model_name, hl, tokenizer, image_processor, context_len, query, image):
    sep = ","
    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 1024
    conv_mode = None
    
    disable_torch_init()

    tokenizer, model, image_processor, context_len = tokenizer, model, image_processor, context_len

    hl = hl

    hl.reinit()

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = [image]
    images = [image.convert('RGB') if image.mode != 'RGB' else image for image in images]
        
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [
        KeywordsStoppingCriteria(keywords, tokenizer, input_ids),
        MaxNewTokensCriteria(input_ids.shape[1], max_new_tokens)
    ]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    attention_output = hl.finalize()
    attention_output = attention_output.view(attention_output.shape[0],24,24)
    attention_output = attention_output.detach()

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )

    # outputs = tokenizer.batch_decode(
    #     output_ids[:, input_token_len:].cpu(), skip_special_tokens=True
    # )[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[: -len(stop_str)]
    # outputs = outputs.strip()
    output = tokenizer.decode(output_ids[:, input_token_len:].cpu()[0])

    token_mapping = get_token_mapping(tokenizer, output, output_ids[:, input_token_len:].cpu()[0])

    return output, {"llava_attentions":attention_output.detach(), "llava_token_mapping":token_mapping}

def clean_text(text):
    cleaned_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', text)
    return cleaned_text

def get_token_mapping(tokenizer, outputs, output_ids):
    tokens = tokenizer.tokenize(outputs)[1:]
    assert len(tokens) == len(output_ids) 
    current_position = 0
    offsets = []

    for token in tokens:
        cleaned_token = clean_text(token)
        try:
            token_start = outputs.find(cleaned_token, current_position)
        except:
            print(outputs, cleaned_token)
            continue
        token_end = token_start + len(cleaned_token)
        offsets.append((token_start, token_end))
        current_position = token_end

    return offsets

def from_preanswer_to_mask(highlight_text, query, cache_dict):
    if highlight_text.strip() == query.strip() or highlight_text.strip() == "":
        token_start_index = 0
        token_end_index = len(cache_dict["llava_token_mapping"]) - 1
    else:
        text_start_index = query.find(highlight_text)
        text_end_index = text_start_index + len(highlight_text)

        for token_index, (token_text_mapping_st, token_text_mapping_end) in enumerate(cache_dict["llava_token_mapping"]):
            if token_text_mapping_st <= text_start_index:
                token_start_index = token_index
            if token_text_mapping_end >= text_end_index:
                token_end_index = token_index
                break
    
    attentions = cache_dict["llava_attentions"]
    selected_attentions = attentions[token_start_index:token_end_index+1]
    mask = selected_attentions.mean(dim=0)
    return mask

def get_model(model_path = "llava-v1.5-7b", device = "cuda:0"):
    model_path = f"liuhaotian/{model_path}"
    model_path = model_path
    model_base = None
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device= device,
        # load_4bit = True,
    )
    return tokenizer, model, image_processor, context_len, model_name

if __name__ == "__main__":
    prompt = "What are the things I should be cautious about when I visit here?"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(BytesIO(requests.get(image_file).content)).convert("RGB")

    tokenizer, model, image_processor, context_len, model_name = get_model()
    
    from .hook import hook_logger
    hl = hook_logger(model, model.device, layer_index = 20)
    output, cache_dict = get_preanswer(model, model_name, hl, tokenizer, image_processor, context_len, prompt, image)

    mask = from_preanswer_to_mask(output[10:20], output, cache_dict)