"""
    Description: gpt4v agent by Azure OpenAI Service
    Author: WangHaotian
    Date: 2023/12/21 14:39
"""
import time
import requests
import torch.nn as nn
from utils.utils import encode_image


class GPT4V(nn.Module):
    def __init__(self, name, args):
        super(GPT4V, self).__init__()
        self.name = name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.time_out = args.time_out
        u = "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version=2023-12-01-preview"
        self.gpt4v_url = u.format(args.gpt4v_resource_name, args.gpt4v_deployment)
        self.gpt4v_header = {"Content-Type": "application/json", "api-key": args.gpt4v_api_key}
        self.error_code = ["401", "429", "500", "503"]

    def generate_answer(self, messages):
        try:
            message_json = {
                "temperature": self.temperature,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
            start_time = time.time()
            chat_response = requests.post(url=self.gpt4v_url, headers=self.gpt4v_header, json=message_json, verify=False, timeout=self.time_out)
            end_time = time.time()
            print(end_time - start_time)
        except Exception as ex:
            print("Find error: {}\nSleep 20s ...".format(ex))
            time.sleep(20)
            return self.generate_answer(messages)

        chat_response = chat_response.json()

        if "error" in chat_response and chat_response["error"]["code"] in self.error_code:
            print("Error Code: {}, Error Message: {}, Sleep 10s ...".format(chat_response["error"]["code"],
                                                                            chat_response["error"]["message"]))
            time.sleep(10)
            return self.generate_answer(messages)

        return chat_response

    def forward(self, messages):
        completion = self.generate_answer(messages)
        return completion


if __name__ == "__main__":
    gpt4v_agent = GPT4V("Test")

    image_path = "../data/test_image.png"
    instruction = "Who are you?"
    base64_image = encode_image(image_path)

    messages = []
    messages.append({"role": "user", "content": []})
    messages[0]["content"].append({"type": "text", "text": instruction})
    messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    completion = gpt4v_agent(messages)
    print(completion["choices"][0]["message"]["content"])

