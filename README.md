# M2D-DKTL

This is the official implementation of the paper "**M2D-DKTL: Multi-modal Multi-agent Debate-based Dialectical Knowledge Transfer Learning**".

## Running Experiments

### Step 0: Preparation
<!-- 首先你需要安装下面的依赖库。 -->
First, you need to install the following additional packages.

```
torch
openai
transformers
```

<!-- 然后你需要下载对应的数据集MMMU、MathVista和CMMMU对应的图片到对应的文件夹。 -->
Next, you need to download the corresponding datasets (MMMU, MathVista, and CMMMU) and the associated images to the appropriate folders.

[[🤗 MMMU](https://huggingface.co/datasets/MMMU/MMMU)] [[🤗 MathVista](https://huggingface.co/datasets/AI4Math/MathVista)] [[🤗 CMMMU](https://huggingface.co/datasets/m-a-p/CMMMU)]

```
|-- data
    |-- images
        |-- MMMU_images
        |-- MathVista_images
        |-- CMMMU_images
```

### Step 1: Multi-modal Multi-agent Debate
<!-- 我们使用了Azure OpenAI提供的API接口，你需要将option.py中的RESOURCE_NAME和API_KEY替换为你自己的。最后运行下面的代码进行多模态多智能体辩论。 -->
We used the API provided by Azure OpenAI. You need to replace the _RESOURCE_NAME_ and _API_KEY_ in `option.py` with your own. Then, run the following code to conduct a multimodal multi-agent debate.

```
sh scripts/run_multi_agent_debate_mmmu.sh         # MMMU dataset
sh scripts/run_multi_agent_debate_math_vista.sh   # MathVista dataset
```

<!-- 由于MMMU和MathVista中的数据有多选选择题和问答题，因此我们在得到辩论结果后使用GPT4对结果进行评估。你可以运行下面的代码对数据进行评估。 -->
Due to the multiple-choice and question-answering tasks in MMMU and MathVista datasets, we use GPT-4 to evaluate the results after obtaining the debate results. You can run the following code to evaluate the data.

```
sh scripts/run_evaluation.sh
```

### Step 2: Construct Chain-of-Debate Data
<!-- 在得到多智能体的辩论过程之后，你可以运行下面的代码从辩论结果中构造出链式辩论数据。 -->
After obtaining the multi-agent debate process, you can run the following code to construct chain-of-debate data from the debate results.

```
python construct_cod_data.py
```

### Step 3: Train Dialectical Knowledge Transfer Learning
<!-- 我们已经将我们基于MMMU数据集凝练好的CoD数据放在了文件夹data/cod_data/下，你可以直接运行下面的代码训练LLaVA，请确保你已经下载了LLaVA原始的模型权重。 -->
We have already placed the CoD data distilled from the MMMU dataset in the folder data/cod_data/. You can directly run the following code to train LLaVA. Please ensure that you have already downloaded the original LLaVA-1.5 model weights [[🤗 LLaVA-1.5](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e)].

```
sh scripts/train_llava_full_7b_13b.sh
```

<!-- 在训练结束后，你可以运行下面的代码对模型进行推理，推理结果可以使用Step 1中的评估代码进行评估。 -->
After the training is complete, you can run the following code for inference. The inference results can be evaluated using the evaluation code from Step 1.

```
sh scripts/inference_llava.sh
```

## Acknowledgement
<!-- 我们基于LLaVA的源码进行了修改，感谢LLaVA的开源。 -->
[LLaVA](https://github.com/haotian-liu/LLaVA): We made modifications based on the source code of LLaVA, and we would like to thank LLaVA for their open-source contributions.