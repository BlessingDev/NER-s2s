
python /workspace/ner_generation/generation_s2s_train.py \
    --model_checkpoint google/flan-t5-base \
    --output_dir /workspace/model_dir/flan-t5-base/ner-json-mix_fp32_encoder_w1e3 \
    --train_epochs 20 \
    --weight_decay 0.001 \
    --batch_size 32 \
    --encoder_weight /workspace/model_dir/flan-t5-base/binary-ner-mixed_fp32_lr1e-5/final_model \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --dataset_name mix \
    --train_file /workspace/datas/few-nerd/supervised/train.preprocessed.mix.csv \
    --validation_file /workspace/datas/few-nerd/supervised/dev.preprocessed.mix.csv