import argparse
import torch
import os
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
import json
from PIL import Image
from tqdm import tqdm

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

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="../llava-v1.5-7b")
parser.add_argument("--model-base", type=str, default=None)
# parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true", default=False)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--image-aspect-ratio", type=str, default='pad')
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument('--source_image_path', type=str, required=True)
parser.add_argument('--source_data', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--find_with_name', action='store_true')
parser.add_argument('--prompt_type', type=str, default='normal', choices=['normal', 'cot', 'self_consistency'])
args = parser.parse_args()

print('load-4bit', args.load_4bit)
print('load-8bit', args.load_8bit)
disable_torch_init()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name,
    # mm_vision_tower='models/source/openai/clip-vit-large-patch14-336'
)
if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print(
        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
            conv_mode, args.conv_mode, args.conv_mode
        )
    )
else:
    args.conv_mode = conv_mode


def predict(query, image_files):
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

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

# predict('What is in these two images?', ['images/llava_example_cmp.png', 'images/llava_logo.png'])
source_image_path = args.source_image_path
image_path = os.listdir(source_image_path)
def find_image(image_id):
    for i in image_path:
        if image_id == int(re.findall(r".*val2014_(.*).jpg", i)[0]):
            return os.path.join(source_image_path, i)
    return "NOT FIND"

def find_image_with_name(image_name):
    if isinstance(image_name, list):
        ret = []
        for image in image_name:
            for i in image_path:
                if image == i:
                    ret.append(os.path.join(source_image_path, i))
        return ret
    else:
        for i in image_path:
            if image_name == i:
                return os.path.join(source_image_path, i)
    print(f'NOT FIND {image_name}')
    exit()
    return 'NOT FIND'


def get_question(data):
    """Return different type question (Open/Multi-Choice)"""
    if data["question_type"] == "multi_choice":
        question = data["question"] + "\nChoices:\n"
        for i, choice in enumerate(data["choices"]):
            question += "({}) {}\n".format(chr(ord("A") + i), choice)
    else:
        question = data["question"]

    return question


cot_prompt = """INSTRUCTION: Thoroughly analyze the information presented in the image. Answer the question step by step as accurately as possible. Put the final answer in the square brackets at the end of your response.

QUESTION: {question}
ANSWER:
"""
normal_prompt = """INSTRUCTION: Answer the question with a word, phrase, or choice.

QUESTION: {question}
ANSWER:"""

all_data = json.load(open(args.source_data, encoding='utf8'))
print(args.__dict__)
out = []
tqdm_len = int(len(all_data)/300)
for data in tqdm(all_data, miniters=tqdm_len):
    question = get_question(data)
    if args.prompt_type == "normal":
        inp = normal_prompt.format(question=question)
    elif args.prompt_type == "cot":
        inp = cot_prompt.format(question=question)
    else:
        inp = normal_prompt.format(question=question)
    # print("inp is: {}".format(inp))
    if args.find_with_name:
        image_file = find_image_with_name(data['image_name'])
    else:
        image_file = find_image(data['image_id'])

    if args.prompt_type == "self_consistency":
        sc_num = 3
    else:
        sc_num = 1

    out_data = data.copy()
    out_data["input"] = inp
    out_data['response'] = []
    for i in range(sc_num):
        outputs = predict(inp, image_file)
        out_data['response'].append(outputs.replace('</s>', ''))
    out.append(out_data)
json.dump(out, open(args.output_path, 'w', encoding='utf8'), indent=2)

print('FINISH',  args.source_data, args.output_path)
