"""
    Description: evaluate the result of multi-agent debate and knowledge transfer
    Author: WangHaotian
    Date: 2024/1/10 11:36
"""

import warnings
import argparse
from tqdm import tqdm

from models.gpt4_agent import GPT4
from utils.utils import read_json_data, read_jsonl_data, get_assistant_message, write_json_data, get_question, \
    get_answer, extract_answer

warnings.filterwarnings("ignore")


def evaluate_dataset(predict_path, eval_agent, eval_prompt, eval_result_path):
    """evaluate math-vista and mmmu dataset"""
    if predict_path.endswith("jsonl"):
        predict_results = read_jsonl_data(predict_path)
    elif "single_agent" in predict_path or "distill" in predict_path or "baseline" in predict_path:
        predict_results = read_json_data(predict_path)
    else:
        predict_results = read_json_data(predict_path)["debate_process"]

    eval_acc = {"all": []}
    eval_results = {"eval_results": [], "eval_metric": {}}
    for result in tqdm(predict_results, total=len(predict_results)):
        instance = result
        data_domain = result["data_domain"]
        question = result["question"]
        truth_answer = get_answer(result)
        llm_answer = result["llm_answer"]

        eval_message = [{"role": "user", "content": eval_prompt.format(question=question,
                                                                       reference_answer=truth_answer,
                                                                       llm_answer=llm_answer)}]
        completion = eval_agent(eval_message)
        agent_eval_result = get_assistant_message(completion)["content"]

        if data_domain not in eval_acc:
            eval_acc[data_domain] = []

        if "[True]" in agent_eval_result:
            eval_acc["all"].append(1.0)
            eval_acc[data_domain].append(1.0)
        else:
            eval_acc["all"].append(0.0)
            eval_acc[data_domain].append(0.0)
        instance["GPT4 Eval Instruction"] = eval_prompt.format(question=question,
                                                               reference_answer=truth_answer,
                                                               llm_answer=llm_answer)
        print(instance["GPT4 Eval Instruction"])
        instance["GPT4 Eval Result"] = agent_eval_result
        eval_results["eval_results"].append(instance)

    print("======= GPT4 Eval Result =======")
    for eval_key in eval_acc:
        eval_results["eval_metric"][eval_key.title()] = "{}/{} = {}".format(sum(eval_acc[eval_key]), len(eval_acc[eval_key]), sum(eval_acc[eval_key]) / len(eval_acc[eval_key]))
        print("[{}] acc is: {}/{} = {}".format(eval_key.title(), sum(eval_acc[eval_key]), len(eval_acc[eval_key]), sum(eval_acc[eval_key]) / len(eval_acc[eval_key])))
    print("======= ====================== =======")
    write_json_data(eval_result_path, eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMMAD Result")
    parser.add_argument("--predict_result_path", type=str,
                        default="results/predictions/math_vista/testmini_500_gpt4v_a2r3.json")
    parser.add_argument("--eval_result_path", type=str,
                        default="results/evaluations/math_vista/testmini_500_gpt4v_a2r3_eval.json")

    parser.add_argument("--gpt4_resource_name", type=str, default="YOUR_RESOURCE_NAME")
    parser.add_argument("--gpt4_api_key", type=str, default="YOUR_API_KEY")
    parser.add_argument("--gpt4_deployment", type=str, default="GPT-4-Turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--time_out", type=int, default=120)
    args = parser.parse_args()

    eval_prompt = """You are a judge agent, your primary responsibility is to impartially evaluate the evaluation answer for correct. Please judge the correctness of the llm answer based on the reference answer. Return [True] if the llm answer matches one of the reference answer, otherwise return [False]. Make sure that True and False are placed in brackets.

    Question: {question}
    Reference Answer: {reference_answer}
    LLM Answer: {llm_answer}
    Judge Answer:"""

    gpt4_agent = GPT4("Evaluation", args)

    evaluate_dataset(args.predict_result_path, gpt4_agent, eval_prompt, args.eval_result_path)
