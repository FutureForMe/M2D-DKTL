"""
    Description: multi-agent debate
    Author: WangHaotian
    Date: 2023/12/22 19:51
"""

import re
import warnings
from tqdm import tqdm

from options import get_args
from models.gpt4v_agent import GPT4V
from models.gpt4_agent import GPT4
from utils.utils import print_args, encode_image, read_json_data, write_json_data, get_assistant_message, extract_answer, \
    get_text_message, get_image_message, get_question
import prompts.prompt as prompt

warnings.filterwarnings("ignore")


def judge_consistency(judge_agent, question, all_agent_answer, judge_prompt):
    """Judge whether the agents response consistency"""
    judge_instruction = judge_prompt.format(question=question, all_answers_from_agents=all_agent_answer)
    judge_instance = [{"role": "user", "content": judge_instruction}]
    completion = judge_agent(judge_instance)
    judge_response = get_assistant_message(completion)["content"]

    pattern = re.compile((r"\[(.*?)\]"))
    result = re.findall(pattern, judge_response)

    if "Yes" in result:
        return True, judge_instruction, judge_response
    else:
        return False, judge_instruction, judge_response


def summary_final_answer(summary_agent, all_agent_answer, question, summary_prompt):
    """Generate the final answer based on the agent response"""
    summary_instruction = summary_prompt.format(question=question, all_answers_from_agents=all_agent_answer)
    summary_instance = [{"role": "user", "content": summary_instruction}]

    completion = summary_agent(summary_instance)
    summary_response = get_assistant_message(completion)

    return summary_instruction, summary_response


if __name__ == "__main__":
    args = get_args()
    print_args(args)

    test_dataset = read_json_data(args.text_data_path)
    error_ids = read_json_data(args.error_ids_path)

    Debaters = [GPT4V("Agent_" + str(i), args) for i in range(args.agent_num)]
    Judge = GPT4("Judge", args)
    Summarizer = GPT4("Summarizer", args)

    # fw_result = open(args.test_output_path[:-5]+".jsonl", "a+", encoding="utf-8")
    all_dialog_process = []
    for test_data in tqdm(test_dataset, total=len(test_dataset)):
        question = get_question(test_data)
        question_id = test_data["question_id"]
        # if question_id not in error_ids:
        #     continue
        answer = test_data["answer"]
        image_paths = [args.image_data_path + image_name for image_name in test_data["image_name"]]
        base64_images = [encode_image(image_path) for image_path in image_paths]

        all_dialog_process.append({"data_domain": test_data["data_domain"],
                                   "question_id": test_data["question_id"],
                                   "question_type": test_data["question_type"],
                                   "question": question,
                                   "choices": test_data["choices"],
                                   "answer": answer,
                                   "image_path": image_paths})

        agents_dialog_history = {}
        for agent in Debaters:
            agents_dialog_history[agent.name] = []

        # Debate
        for round_num in range(args.max_round):
            all_agent_answer_judge, all_agent_answer_summary = "", ""
            all_dialog_process[-1]["stop_round"] = round_num
            all_dialog_process[-1]["Round_{}".format(round_num)] = {}

            for num in range(args.agent_num):
                debater_name = Debaters[num].name
                debater_messages = []
                if round_num == 0:
                    debate_prompt = prompt.debater_first_round_prompt.format(question=question)

                else:
                    agent_historical_answer = "HERE IS YOUR HISTORICAL ANSWER:\n" + agents_dialog_history[debater_name][-1]["content"] + "\n(END OF YOUR HISTORICAL ANSWER)"
                    other_agent_historical_answer = "ANSWERS FROM OTHER AGENTS:\n"
                    for agent_name in agents_dialog_history:
                        if agent_name == debater_name:
                            continue
                        other_agent_historical_answer += "({}) {}\n(END OF {} ANSWER)\n".format(agent_name, agents_dialog_history[agent_name][-1]["content"], agent_name)

                    debate_prompt = prompt.debater_other_round_prompt.format(question=question,
                                                                             answer_from_other_agents=other_agent_historical_answer,
                                                                             your_historical_answer=agent_historical_answer)

                debater_messages.append({"role": "user", "content": []})
                debater_messages[0]["content"].extend(get_text_message(debate_prompt))
                debater_messages[0]["content"].extend(get_image_message(base64_images, args))

                agents_dialog_history[debater_name].extend(debater_messages.copy())
                completion = Debaters[num](debater_messages)
                response = get_assistant_message(completion)
                agents_dialog_history[debater_name].append(response)

                all_dialog_process[-1]["Round_{}".format(round_num)]["User_{}".format(num)] = debate_prompt
                all_dialog_process[-1]["Round_{}".format(round_num)]["Agent_{}".format(num)] = response["content"]
                all_agent_answer_judge += "(Agent {}): {}\n".format(num, extract_answer(response["content"]))
                all_agent_answer_summary += "(Agent {}): {}\n".format(num, response["content"])

            # Judge
            judge_result, judge_instruction, judge_response = judge_consistency(Judge, question, all_agent_answer_judge, prompt.judge_prompt)
            all_dialog_process[-1]["Round_{}".format(round_num)]["Judge_Instruction"] = judge_instruction
            all_dialog_process[-1]["Round_{}".format(round_num)]["Judge_Response"] = judge_response

            if judge_result or round_num == args.max_round - 1:
                if round_num < args.max_round - 1:
                    print("Early Stop!")

                # Summary
                summary_instruction, summary_result = summary_final_answer(Summarizer, all_agent_answer_summary, question, prompt.summary_prompt)
                all_dialog_process[-1]["Summary"] = {}
                all_dialog_process[-1]["Summary"]["User"] = summary_instruction
                all_dialog_process[-1]["Summary"]["Assistant"] = summary_result["content"]
                all_dialog_process[-1]["llm_answer"] = extract_answer(summary_result["content"])
                break
        print(all_dialog_process[-1])

    final_result = {"args": args.__dict__, "debate_process": all_dialog_process}
    write_json_data(args.test_output_path, final_result)


