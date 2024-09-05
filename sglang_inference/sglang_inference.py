import os, argparse, json, datetime, time, sys
from PIL import Image
import torch

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text

from DatasetManager.dataloader import get_data

GEN_KWARGS = {
    ('textVQA', 'VisWiz', 'MMMU', 'MME', "POPE", "POPE3k"): {"max_tokens": 16},
    ('mmvet', 'LLaVA_Wild'): {"max_tokens": 2048},
    }
COMMON_KWARGS = {"temperature": 0.0}

def get_gen_kwargs(dataset):
    for datasets, kwargs in GEN_KWARGS.items():
        if dataset in datasets:
            kwargs.update(COMMON_KWARGS)
            return kwargs
    raise NotImplementedError

@sgl.function
def image_qa(s, image_file, question):
    s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_tokens))

def genquery(prompt, question, previous_answer):
    question = '"' + question +'"'
    previous_answer = '"' + previous_answer +'"'
    prompt = prompt.replace("[R]", previous_answer)
    prompt = prompt.replace("[Q]", question)
    return prompt

def get_images_path(original_image_path, image_folder, key, image_type):
    if image_type == "original":
        image_path =  original_image_path
    elif image_type == "masked":
        image_path = os.path.join(image_folder, f"{key}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, f"{key}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        raise NotImplementedError            
    return image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mmvet")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_folder", type=str, default="../some/images")
    parser.add_argument("--prompt_json", type=str, default="../API/Prompts/prompts.json")
    parser.add_argument("--prompt_name", type=str, default="test")
    parser.add_argument("--port_value", type=int, default=30000)
    parser.add_argument("--output_folder", type=str, default="../results/llava15")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--previous_answer_file", type=str, default="")
    args = add_common_sglang_args_and_parse(parser)
    args.port=args.port_value

    for k,v in get_gen_kwargs(args.dataset).items():
        setattr(args, k, v)

    dataset = get_data(args.dataset, formated = True, forced_image_path = True)

    with open(args.prompt_json, "r") as json_file:
        prompts = json.load(json_file)
        prompt = prompts[args.prompt_name]

    results = {"args": vars(args), "answers": {}}
    result_path = os.path.join(f"{args.output_folder}", args.exp_name + ".json")
    print("output file: ", os.path.abspath(result_path))
    os.makedirs(os.path.dirname(result_path), exist_ok = True)

    arguments = []

    for index in range(len(dataset)):
        key, image_path, question = dataset.get_item(index)
        if args.previous_answer_file:
            previous_answer = previous_answers[key] if key in previous_answers else ""
        else:
            previous_answer = ""
        if key in results["answers"]:
            continue
        else:
            # first round
            prompt_item = prompt[0]
            image_path = get_images_path(image_path, args.image_folder, key, prompt_item[1])
            query = genquery(prompt_item[0], question, previous_answer)
            # print(query)
            arguments.append(
                {"image_file": image_path, "question": query}
                )

    states = [None] * len(arguments)

    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    tic = time.time()
    states = image_qa.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=False)
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    for index in range(len(dataset)):
        key, image_path, question = dataset.get_item(index)
        if key in results["answers"]:
            continue
        else:
            results["answers"][key] = states.pop(0)["answer"].strip()
        
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Finished {args.exp_name} @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")