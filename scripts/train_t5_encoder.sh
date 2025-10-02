
learning_rates=(1e-5)

for lr in ${learning_rates[@]}
do
    echo "Training with learning rate: $lr"
    python /workspace/encoder_binary/encoder_binary_train.py \
        --model_checkpoint google/flan-t5-base \
        --output_dir /workspace/model_dir/flan-t5-base/binary-ner-mixed_fp32_lr${lr} \
        --weight_decay 0.01 \
        --gradient_accumulation_steps 1 \
        --batch_size 128 \
        --learning_rate $lr \
        --num_epochs 20
done