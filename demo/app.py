import os
import gradio as gr
import torch
import base64
import requests
from io import BytesIO

from API_LLaVA.functions import get_model as llava_get_model, get_preanswer as llava_get_preanswer, from_preanswer_to_mask as llava_from_preanswer_to_mask
from API_LLaVA.hook import hook_logger as llava_hook_logger
from API_LLaVA.main import blend_mask as llava_blend_mask

from API_CLIP.main import get_model as clip_get_model, gen_mask as clip_gen_mask, blend_mask as clip_blend_mask

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MARKDOWN = """
<div align='center'>
    <b style="font-size: 2em;">API: Attention Prompting on Image for Large Vision-Language Models</b>
    <br>    
    <br>    
    <br>    
    [<a href="https://arxiv.org/abs/"> arXiv paper </a>] 
    [<a href="https://yu-rp.github.io/api-prompting/"> project page </a>]
    [<a href="https://pypi.org/project/apiprompting/"> python package </a>]
    [<a href="https://github.com/yu-rp/apiprompting"> code </a>]
</div>
"""

def get_base64_images(image):
    image = image.convert('RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

def vqa(image, question, api_key):
    base64_image = get_base64_images(image)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-turbo-2024-04-09",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": question
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail":"low"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def compare(input_image, output_image, query, api_key):
    original_response = vqa(input_image, query, api_key)
    api_response = vqa(output_image, query, api_key)
    return original_response, api_response

def init_clip():
    clip_model, clip_prs, clip_preprocess, _, clip_tokenizer = clip_get_model(
        model_name = "ViT-L-14-336" if torch.cuda.is_available() else "ViT-L-14", 
        layer_index = 22, device= DEVICE)
    return {"clip_model": clip_model, "clip_prs": clip_prs, "clip_preprocess": clip_preprocess, "clip_tokenizer": clip_tokenizer}

def init_llava():
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_model_name = llava_get_model("llava-v1.5-13b", device= DEVICE)
    llava_hl = llava_hook_logger(llava_model, DEVICE, layer_index = 20)
    return {"llava_tokenizer": llava_tokenizer, "llava_model": llava_model, "llava_image_processor": llava_image_processor, "llava_context_len": llava_context_len, "llava_model_name": llava_model_name, "llava_hl": llava_hl}

def change_api_method(api_method):
    new_text_pre_answer = gr.Textbox(
        label="LLaVA Response",
        info = 'Only used for LLaVA-Based API. Press "Pre-Answer" to generate the response.',
        placeholder="",
        value = "",
        lines=4,
        interactive=False,
        type="text")
    new_image_output = gr.Image(
            label="API Masked Image",
            type="pil",
            interactive=False,
            height=512
        )
    if api_method == "CLIP_Based API":
        model_dict = init_clip()
        new_generate_llava_response_button = gr.Button("Pre-Answer", interactive=False)
    elif api_method == "LLaVA_Based API":
        model_dict = init_llava()
        new_generate_llava_response_button = gr.Button("Pre-Answer", interactive=True)
    else:
        raise NotImplementedError
    return model_dict, {}, new_generate_llava_response_button, new_text_pre_answer, new_image_output

def clear_cache(cache_dict):
    return {}

def clear_mask_cache(cache_dict):
    if "llava_mask" in cache_dict.keys():
        del cache_dict["llava_mask"] 
    if "clip_mask" in cache_dict.keys():
        del cache_dict["clip_mask"] 
    return cache_dict

def llava_pre_answer(image, query, cache_dict, model_dict):
    pre_answer, cache_dict_update = llava_get_preanswer(
        model_dict["llava_model"], 
        model_dict["llava_model_name"], 
        model_dict["llava_hl"], 
        model_dict["llava_tokenizer"], 
        model_dict["llava_image_processor"], 
        model_dict["llava_context_len"], 
        query, image)
    cache_dict.update(cache_dict_update)
    return pre_answer, cache_dict

def generate_mask(
        image, 
        query, 
        pre_answer,
        highlight_text,
        api_method,
        enhance_coe,
        kernel_size,
        interpolate_method_name,
        mask_grayscale,
        cache_dict,
        model_dict):
    if api_method == "LLaVA_Based API":
        assert highlight_text.strip() in pre_answer
        if "llava_mask" in cache_dict.keys() and cache_dict["llava_mask"] is not None:
            pass
        else:
            cache_dict["llava_mask"] = llava_from_preanswer_to_mask(highlight_text, pre_answer, cache_dict)
        masked_image = llava_blend_mask(image, cache_dict["llava_mask"], enhance_coe, kernel_size, interpolate_method_name, mask_grayscale)
    elif api_method == "CLIP_Based API":
        # assert highlight_text in query
        if "clip_mask" in cache_dict.keys() and cache_dict["clip_mask"] is not None:
            pass
        else:
            cache_dict["clip_mask"] = clip_gen_mask(
                model_dict["clip_model"], 
                model_dict["clip_prs"], 
                model_dict["clip_preprocess"], 
                DEVICE, 
                model_dict["clip_tokenizer"], 
                [image], 
                [highlight_text if highlight_text.strip() != "" else query])
        masked_image = clip_blend_mask(image, *cache_dict["clip_mask"], enhance_coe, kernel_size, interpolate_method_name, mask_grayscale)
    else:
        raise NotImplementedError
    return masked_image, cache_dict


image_input = gr.Image(
    label="Input Image",
    type="pil",
    interactive=True,
    height=512
)
image_output = gr.Image(
    label="API Masked Image",
    type="pil",
    interactive=False,
    height=512
)

text_query = gr.Textbox(
    label="Query",
    placeholder="Enter a query about the image",
    lines=2,
    type="text")
text_pre_answer = gr.Textbox(
    label="LLaVA Response",
    info = 'Only used for LLaVA-Based API. Press "Pre-Answer" to generate the response.',
    placeholder="",
    lines=2,
    interactive=False,
    type="text")
text_highlight_text = gr.Textbox(
    label = "Hint Text.",
    info = "The text based on which the mask will be generated. For LLaVA-Based API, it should be a substring of the pre-answer.",
    placeholder="Enter the hint text",
    lines=1,
    type="text")
text_api_token = gr.Textbox(
    label = "OpenAI API Token",
    placeholder="Input your OpenAI API token",
    lines=1,
    type="text")
text_original_image_response = gr.Textbox(
    label="GPT Response (Original Image)",
    placeholder="",
    lines=2,
    interactive=False,
    type="text")
text_API_image_response = gr.Textbox(
    label="GPT Response (API-maksed Image)",
    placeholder="",
    lines=2,
    interactive=False,
    type="text")

radio_api_method = gr.Radio(
    ["CLIP_Based API", "LLaVA_Based API"] if torch.cuda.is_available() else ["CLIP_Based API"], 
    interactive=True,
    value = "CLIP_Based API",
    label="Type of API")
slider_mask_grayscale = gr.Slider(
    minimum=0,
    maximum=255,
    step = 0.5,
    value=100,
    interactive=True,
    info = "0: black mask, 255: white mask.",
    label="Grayscale")
slider_enhance_coe = gr.Slider(
    minimum=1,
    maximum=50,
    step = 1,
    value=1,
    interactive=True,
    info = "The larger contrast, the greater the contrast between the bright and dark areas of the mask.",
    label="Contrast")
slider_kernel_size = gr.Slider(
    minimum=1,
    maximum=9,
    step = 2,
    value=1,
    interactive=True,
    info = "The larger smoothness, the smoother the mask appears, reducing the rectangular shapes.",
    label="Smoothness")
radio_interpolate_method_name = gr.Radio(
    ["BICUBIC",  "BILINEAR","BOX","LANCZOS", "NEAREST"], 
    value = "BICUBIC",
    interactive=True,
    label="Interpolation Method", 
    info="The interpolation method used during mask resizing.")

generate_llava_response_button = gr.Button("Pre-Answer", interactive=False)
generate_mask_button = gr.Button("API Go!")
ask_gpt_button = gr.Button("GPT Go!")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    state_cache = gr.State({})
    state_model = gr.State(init_clip())
    with gr.Row():
        image_input.render()
        image_output.render()    
    with gr.Accordion("Query and API Processing"):       
        with gr.Row():
            radio_api_method.render()
        with gr.Row(equal_height=True):
            with gr.Column():
                text_query.render()
                generate_llava_response_button.render()
                text_pre_answer.render()
                text_highlight_text.render()
            with gr.Column():
                slider_enhance_coe.render()
                slider_kernel_size.render()
                radio_interpolate_method_name.render()
                slider_mask_grayscale.render()
        with gr.Row():
            generate_mask_button.render()
    with gr.Accordion("GPT Response"):      
        text_api_token.render()
        ask_gpt_button.render()
        with gr.Row():
            text_original_image_response.render()
            text_API_image_response.render()
    with gr.Accordion("Examples"):  
        examples_images_responses = gr.Examples(
            [

            ],
            [
                image_input, 
                image_output, 
                text_query, 
                text_pre_answer, 
                text_highlight_text, 
                slider_enhance_coe,
                slider_kernel_size,
                radio_interpolate_method_name,
                slider_mask_grayscale,
                text_original_image_response,
                text_API_image_response
                ],
            )

    radio_api_method.change(
        fn=change_api_method,
        inputs = [radio_api_method],
        outputs=[state_model, state_cache, generate_llava_response_button, text_pre_answer, image_output]
    )

    image_input.change(
        fn=clear_cache,
        inputs = state_cache,
        outputs=state_cache
    )
    text_query.change(
        fn=clear_cache,
        inputs = state_cache,
        outputs=state_cache
    )
    text_highlight_text.change(
        fn=clear_mask_cache,
        inputs = state_cache,
        outputs=state_cache
    )

    generate_llava_response_button.click(
        fn=llava_pre_answer,
        inputs=[image_input, text_query, state_cache, state_model],
        outputs=[text_pre_answer, state_cache]
    )
    generate_mask_button.click(
        fn=generate_mask,
        inputs=[
            image_input, 
            text_query, 
            text_pre_answer,
            text_highlight_text,
            radio_api_method,
            slider_enhance_coe,
            slider_kernel_size,
            radio_interpolate_method_name,
            slider_mask_grayscale,
            state_cache,
            state_model
            ],
        outputs=[image_output, state_cache]
    )
    ask_gpt_button.click(
        fn=compare,
        inputs=[image_input, image_output, text_query, text_api_token],
        outputs=[text_original_image_response, text_API_image_response]
    )

demo.queue(max_size = 1).launch(show_error=True)