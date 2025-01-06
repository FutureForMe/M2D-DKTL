"""
    Description: gpt4 agent by Azure OpenAI Service
    Author: WangHaotian
    Date: 2023/12/21 15:21
"""

import time
import requests
import torch.nn as nn


class GPT4(nn.Module):
    def __init__(self, name, args):
        super(GPT4, self).__init__()
        self.name = name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.time_out = args.time_out
        u = "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version=2023-12-01-preview"
        self.gpt4_url = u.format(args.gpt4_resource_name, args.gpt4_deployment)
        self.gpt4_header = {"Content-Type": "application/json", "api-key": args.gpt4_api_key}
        self.error_code = ["401", "429", "500", "503"]

    def generate_answer(self, messages):
        try:
            message_json = {"messages": messages, "temperature": self.temperature}
            start_time = time.time()
            chat_response = requests.post(url=self.gpt4_url, headers=self.gpt4_header, json=message_json, verify=False, timeout=self.time_out)
            end_time = time.time()
            print(end_time - start_time)
        except Exception as ex:
            print("Find Error: {}\nSleep 10s ...".format(ex))
            time.sleep(10)
            return self.generate_answer(messages)

        chat_response = chat_response.json()

        if "error" in chat_response and chat_response["error"]["code"] in self.error_code:
            print("Error Code: {}, Error Message: {}, Sleep 10s ...".format(chat_response["error"]["code"],
                                                                            chat_response["error"]["message"]))
            time.sleep(10)
            return self.generate_answer(messages)

        rate_limit = "have exceeded token rate limit"
        if "choices" in chat_response and rate_limit in chat_response["choices"][0]["message"]["content"]:
            print("Token rate limit error ...")
            time.sleep(10)
            return self.generate_answer(messages)

        return chat_response

    def forward(self, messages):
        completion = self.generate_answer(messages)
        return completion


if __name__ == "__main__":

    gpt4v_agent = GPT4("Test")

    messages = [{"role": "system", "content": "You are an intelligent agent."},
                {"role": "user", "content": "Who are you?"}]

    completion = gpt4v_agent(messages)
    print(completion["choices"][0]["message"]["content"])

