export CUDA_VISIBLE_DEVICES=0

################# Full LLaVA 7B Inference ###################
python inference_llava.py \
    --model-path  ./checkpoints/llava_v1.5/7b/llava_full_CoD/checkpoint-300 \
    --source_image_path ./data/images/MMMU_images/     \
    --source_data ./data/test/MMMU_validation_50_per.json     \
    --output_path ./results/llava_inference/MMMU/Full_llava_7b_CoD_MMMU_50per_checkpiont_300_result.json     \
    --find_with_name



################# Full LLaVA 13B Inference ###################
python inference_llava.py \
    --model-path  ./checkpoints/llava_v1.5/13b/llava_full_CoD/checkpoint-300 \
    --source_image_path ./data/images/MMMU_images/     \
    --source_data ./data/test/MMMU_validation_50_per.json     \
    --output_path ./results/llava_inference/MMMU/Full_llava_13b_CoD_MMMU_50per_checkpiont_300_result.json     \
    --find_with_name