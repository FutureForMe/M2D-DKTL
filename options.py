"""
    Description: arguments for experiments.
    Author: WangHaotian
    Date: 2023/12/22 10:58
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Multi-modal Multi-agent Debate")
    parser.add_argument("--dataset_name", type=str, default="math_vista", choices=["mm_vet", "mmmu", "math_vista"])
    parser.add_argument("--text_data_path", type=str, default="./data/processed/MathVista/math_vista_testmini_500_v1.json")
    parser.add_argument("--image_data_path", type=str,
                        default="/home/nvidia/wht/MathVista_images/images/")
    parser.add_argument("--test_output_path", type=str,
                        default="./results/predictions/mmmu/testmini_500_gpt4v_a2r3_easy_error_result_prompt_v2.1.json")

    parser.add_argument("--data_begin", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=3000)
    parser.add_argument("--gpt4v_deployment", type=str, default="GPT-4-Turbo-Vision")
    parser.add_argument("--gpt4v_resource_name", type=str, default="YOUR_GPT4V_RESOURCE_NAME")
    parser.add_argument("--gpt4v_api_key", type=str, default="YOUR_GPT4V_API_KEY")

    parser.add_argument("--gpt4_deployment", type=str, default="GPT-4-Turbo")
    parser.add_argument("--gpt4_resource_name", type=str, default="YOUR_GPT4_RESOURCE_NAME")
    parser.add_argument("--gpt4_api_key", type=str, default="YOUR_GPT4_API_KEY")

    parser.add_argument("--agent_num", type=int, default=2)
    parser.add_argument("--max_round", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--time_out", type=int, default=120)
    parser.add_argument("--gpt4v_detail", type=str, default="auto", choices=["low", "auto", "high"])
    parser.add_argument("--single_prompt_type", type=str, default="normal", choices=["normal", "cot"])

    args = parser.parse_args()

    return args
