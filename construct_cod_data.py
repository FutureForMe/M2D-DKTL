"""
    Description: construct chain-of-debate data
    Author: WangHaotian
    Date: 2024/1/17 20:08
"""
import json
import argparse
import warnings
from tqdm import tqdm

from models.gpt4v_agent import GPT4V
from utils.utils import read_json_data, encode_image, get_image_message, get_text_message, get_assistant_message, \
    write_json_data

warnings.filterwarnings("ignore")


cod_prompt = """Based on the question and the answers of all agents, extract the reasoning process in the form of a logical and concise chain of thought. Put the final answer in the square brackets.
Here is an example of answer:
ANSWER: LOGICAL CHAIN OF THOUGHT:
1. The sickly, yellowed leaves on the left branch indicate a localized issue, not a systemic one, as it is not affecting the entire tree uniformly.
2. The death of some branches suggests a severe localized problem.
3. Virus infection is less likely as it would typically affect the plant more uniformly.
4. Herbicide damage is a likely cause as it can affect parts of plants differently based on exposure and would explain the localized nature of the damage.
5. Fungus disease often shows more visible signs like spots or growths, which are not mentioned.
6. Phytoplasma would also typically affect the plant more uniformly.
7. The most plausible cause, given the symptoms and their distribution, is herbicide damage.

Final Answer: [(B) Herbicide damage]
(END OF EXAMPLE)

QUESTION: {question}
DEBATE PROCESS:
{debate_process}

ANSWER:"""


if __name__ == "__main__":
    debate_result_path = ""
    parser = argparse.ArgumentParser(description="Construct Chain-of-Debate dataset")
    parser.add_argument("--debate_result_path", type=str,
                        default="./results/mmmu/mmmu_debate_result.json")
    parser.add_argument("--cod_data_path", type=str,
                        default="./results/mmmu/mmmu_cod_data.json")

    parser.add_argument("--gpt4v_deployment", type=str, default="GPT-4-Turbo-Vision")
    parser.add_argument("--gpt4v_resource_name", type=str, default="YOUR_RESOURCE_NAME")
    parser.add_argument("--gpt4v_api_key", type=str, default="YOUR_API_KEY")
    parser.add_argument("--gpt4v_detail", type=str, default="auto")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--time_out", type=int, default=120)
    args = parser.parse_args()

    debate_results = read_json_data(args.debate_result_path)

    gpt4v_agent = GPT4V("Construct_CoD", args)

    fw_cod = open(args.cod_data_path, "a+", encoding="utf-8")

    cod_data = []
    for result in tqdm(debate_results, total=len(debate_results)):
        question = result["question"]

        debate_process = ""
        for i in range(result["stop_round"] + 1):
            round_num = "Round_{}".format(str(i))
            debate_process += "**{}**\n".format(round_num)
            debate_process += "(Agent 0) " + result[round_num]["Agent_0"] + "\n(END OF AGENT 0)\n"
            debate_process += "(Agent 1)" + result[round_num]["Agent_1"] + "\n(END OF AGENT 1)\n"

        image_paths = result["image_path"]
        base64_images = [encode_image(image_path) for image_path in image_paths]

        messages = []
        messages.append({"role": "user", "content": []})
        messages[0]["content"].extend(get_text_message(cod_prompt.format(question=question, debate_process=debate_process)))
        messages[0]["content"].extend(get_image_message(base64_images, args))

        completion = gpt4v_agent(messages)

        single_agent_response = get_assistant_message(completion)["content"]
        result["Chain_of_Debate"] = single_agent_response
        cod_data.append(result)
        print(single_agent_response)

    write_json_data(args.cod_data_path, cod_data)

