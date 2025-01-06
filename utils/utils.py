"""
    Description: utils for experiments
    Author: WangHaotian
    Date: 2023/12/21 15:25
"""
import re
import json
import base64


def encode_image(image_path):
    """encode the image"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def print_args(args):
    """print parameters"""
    for key in args.__dict__:
        print(key + ": " + str(args.__dict__[key]))


def read_json_data(data_path):
    """read json format dataset"""
    print("Load json data from path: {}".format(data_path))
    with open(data_path, "r", encoding="utf-8") as fr:
        test_dataset = json.load(fr)

    return test_dataset


def read_jsonl_data(data_path):
    """read jsonl format dataset"""
    print("Load jsonl data from path: {}".format(data_path))
    with open(data_path, "r", encoding="utf-8") as fr:
        test_dataset = []
        for line in fr.readlines():
            test_dataset.append(json.loads(line))

    return test_dataset


def write_json_data(data_path, data_json):
    """write data to json"""
    print("Write json data to path: {}".format(data_path))
    with open(data_path, "w", encoding="utf-8") as fw:
        json.dump(data_json, fw, indent=4, ensure_ascii=False)


def write_jsonl_data(data_path, data_json):
    """write data to jsonl"""
    print("Write jsonl data to path: {}".format(data_path))
    with open(data_path, "w", encoding="utf-8") as fw:
        for data in data_json:
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")


def get_assistant_message(completion):
    """get response of agent"""
    if "error" in completion:
        return {"role": "assistant", "content": completion["error"]["message"]}

    if "content" not in completion["choices"][0]["message"]:
        completion["choices"][0]["message"]["content"] = ""

    return completion["choices"][0]["message"]


def extract_answer(response):
    """extract answer for []"""
    pattern = re.compile((r"\[(.*?)\]"))
    predictions = re.findall(pattern, response)
    if len(predictions) == 0:
        return "None"
    else:
        return predictions[-1]


def get_text_message(prompt):
    """Return text in message"""
    return [{"type": "text", "text": prompt}]


def get_image_message(base64_images, args):
    """Return based64 image in message"""
    image_content = []
    for base64_image in base64_images:
        image_content.append({"type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": args.gpt4v_detail}
                              })
    return image_content


def get_question(data):
    """Return different type question (Open/Multi-Choice)"""
    if data["question_type"] == "multi_choice":
        question = data["question"] + "\nChoices:\n"
        for i, choice in enumerate(data["choices"]):
            question += "({}) {}\n".format(chr(ord("A") + i), choice)
    else:
        question = data["question"]

    return question


def get_answer(data):
    """Return ground truth(answer)"""
    if data["question_type"] == "multi_choice":
        choice = chr(ord("A") + data["choices"].index(data["answer"]))
        answer = "({}) {}".format(choice, data["answer"])
    else:
        answer = data["answer"]

    return answer
