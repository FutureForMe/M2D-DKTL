#!/bin/bash
#############   LLaVA 1.5 7B Lora #############
deepspeed --include localhost:2,3 --master_port=7001 llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-5 \
    --deepspeed ./scripts/train_llava_zero3.json \
    --model_name_or_path ./liuhaotian/llava-v1.5-7b/ \
    --version v1 \
    --data_path ./data/cod_data/MMMU_a2r3_CoD.json \
    --image_folder ./data/images/MMMU_images/ \
    --vision_tower ./openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_v1.5/7b/llava_lora_CoD \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none


    
    
#############   LLaVA 1.5 13B Lora #############
deepspeed --include localhost:2,3 --master_port=7001 llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-5 \
    --deepspeed ./scripts/train_llava_zero3.json \
    --model_name_or_path ./liuhaotian/llava-v1.5-13b/ \
    --version v1 \
    --data_path ./data/cod_data/MMMU_a2r3_CoD.json \
    --image_folder ./data/images/MMMU_images/ \
    --vision_tower ./openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_v1.5/13b/llava_lora_CoD \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
