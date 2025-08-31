datasets=("abalone" "cardio" "crowdfunding" "flight-price" "gaming" "heart-failure" "housing" "insurance" "student-performance" "weather")

for dataset in "${datasets[@]}"; do
    python finetune.py --stage "SFT" \
        --llm_path "mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
        --train_ds "finetuning_data/50/$dataset/train" \
        --eval_ds "finetuning_data/50/$dataset/eval" \
        --output_dir "finetuned_models/50/$dataset/" \
        --epochs 1
done
