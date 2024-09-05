import os, json, hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import math
import base64
from PIL import Image
import io

from datasets import load_dataset, concatenate_datasets

def get_data(dataset, **kwargs):
    if dataset == 'mmvet':
        return MMVET(**kwargs)
    elif dataset == 'textVQA':
        return textVQA(**kwargs)
    elif dataset == 'VisWiz':
        return VisWiz(**kwargs)
    elif dataset == 'MMMU':
        return MMMU(**kwargs)
    elif dataset == 'LLaVA_Wild':
        return LLaVA_Wild(**kwargs)
    elif dataset == 'MME':
        return MME(**kwargs)
    else:
        raise ValueError('Invalid dataset')

def Disposable_Dataloader(dataset, batch_size, drange, **kwargs):
    for i in range(drange[0], min(len(dataset),drange[1]), batch_size):
        st = i
        ed = i + batch_size
        keys, image_path_or_pil_images, questions = [] , [] , []
        for j in range(st, ed):
            if j >= min(len(dataset),drange[1]):
                break
            key, image_path_or_pil_image, question = dataset.get_item(j)
            keys.append(key)
            image_path_or_pil_images.append(image_path_or_pil_image)
            questions.append(question)
        yield keys, image_path_or_pil_images, questions


MMMU_CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

MME_SUB_1 = ['scene', 'posters', 'celebrity']
MME_SUB_2 = ['count', 'code_reasoning', 'existence', 'OCR', 'color', 'commonsense_reasoning', 'numerical_calculation', 'position', 'text_translation']


def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

class MMVET:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/mm-vet","mm-vet.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)

        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question
        return question

    def get_item(self, index = 0):
        key = f"v1_{index}"
        item = self.json[key]
        image_path = os.path.join("../../dataset/mm-vet","images",item["imagename"])
        question = self.formated_question(item["question"])
        return key, image_path, question

    def __len__(self):
        return len(self.json)

class textVQA:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/TextVQA/TextVQA_0.5.1_val.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)
            self.data = self.json['data']
        
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question + " Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = os.path.join("../../dataset/TextVQA/train_images",item["image_id"]+".jpg")
        question = self.formated_question(item["question"])
        return key, image_path, question

    def __len__(self):
        return len(self.data)
        
class VisWiz:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/VisWiz/val.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)

        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question + " When the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.json[index]
        image_path = os.path.join("../../dataset/VisWiz/val",item["image"])
        question = item["question"]
        question = self.formated_question(question)
        return key, image_path, question

    def __len__(self):
        return len(self.json)

class MMMU:
    def __init__(self, **kwargs):
        data_path = "../../dataset/MMMU/MMMU"
        split = "validation"
        # run for each subject
        sub_dataset_list = []
        for subject in MMMU_CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.dataset = concatenate_datasets(sub_dataset_list)

        self.formated = kwargs.get('formated', False)

    def get_question(self, question, options):
        return question

    def formated_question(self, sample):
        if self.formated:
            question = sample['question']
            options = eval(sample['options'])
            example = ""
            if sample['question_type'] == 'multiple-choice':
                start_chr = 'A'
                prediction_range = []
                index2ans = {}
                for option in options:
                    prediction_range.append(start_chr)
                    example += f"({start_chr}) {option}\n"
                    index2ans[start_chr] = option
                    start_chr = chr(ord(start_chr) + 1)
                empty_prompt_sample_structure = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
                empty_prompt = empty_prompt_sample_structure.format(question, example)
            else:
                empty_prompt_sample_structure = "{}\n\nAnswer the question using a single word or phrase."
                empty_prompt = empty_prompt_sample_structure.format(question)
            return empty_prompt
        else:
            return sample['question']

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.dataset[index]
        question = self.formated_question(item)
        pil_image = item["image_1"]
        return key, pil_image, question

    def __len__(self):
        return len(self.dataset)

class LLaVA_Wild:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/LLaVA_Wild/llava-bench-in-the-wild/questions.jsonl")
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.data = data

        self.formated = kwargs.get('formated', False)

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = os.path.join("../../dataset/LLaVA_Wild/llava-bench-in-the-wild","images",item["image"])
        question = item["text"]
        question = self.formated_question(question)
        return key, image_path, question

    def formated_question(self, question):
        if self.formated:
            question = question
        return question

    def __len__(self):
        return len(self.data)

class MME:
    def __init__(self, **kwargs):
        base_path = "../../dataset/MME/MME_Benchmark_release_version"
        self.data = []
        for sub in MME_SUB_1:
            sub_folder = os.path.join(base_path, sub)
            question_folder = os.path.join(sub_folder, "questions_answers_YN")
            image_folder = os.path.join(sub_folder, "images")
            for question_file in sorted(os.listdir(question_folder)):
                question_file_path = os.path.join(question_folder, question_file)
                image_name = question_file.split('.')[0] + ".jpg"
                image_path = os.path.join(image_folder, image_name)
                if not os.path.exists(os.path.join(base_path, "images", image_name)):
                    continue
                with open(question_file_path, 'r') as file:
                    for l in file.readlines():
                        question, answer = l.strip().split('\t')
                        self.data.append([
                            image_path,
                            question,
                            answer,
                            sub
                        ])
        for sub in MME_SUB_2:
            sub_folder = os.path.join(base_path, sub)
            files = os.listdir(sub_folder)
            file_names = sorted(list(set([f.split('.')[0] for f in files])))
            for file_name in file_names:
                question_file = file_name + ".txt"
                image_name = file_name + ".jpg"
                question_file_path = os.path.join(sub_folder, question_file)
                image_path = os.path.join(sub_folder, image_name)
                if not os.path.exists(question_file_path) or not os.path.exists(image_path):
                    continue
                with open(question_file_path, 'r') as file:
                    for l in file.readlines():
                        question, answer = l.strip().split('\t')
                        self.data.append([
                            image_path,
                            question,
                            answer,
                            sub
                        ])

        self.formated = kwargs.get('formated', False)
            
    def formated_question(self, question):
        if self.formated:
            question = question + " Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = item[0]
        question = item[1]
        question = self.formated_question(question)
        return key, image_path, question

    def __len__(self):
        return len(self.data)
        