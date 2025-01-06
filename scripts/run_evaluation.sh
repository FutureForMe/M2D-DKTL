######## MMMU #########
python evaluation.py \
 --predict_result_path ./results/predictions/mmmu_/validation_50_per_a2r3_result.json \
 --eval_result_path ./results/evaluations/mmmu_/validation_50_per_a2r3_result_eval.json

######## MathVista #########
 python evaluation.py \
 --predict_result_path ./results/predictions/math_vista/testmini_500_gpt4v_a2r3_result.json \
 --eval_result_path  ./results/evaluations/math_vista/testmini_500_gpt4v_a2r3_result_eval.json