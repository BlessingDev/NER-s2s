
learning_rates=(2e-4)
dropout_rates=(0.2)
label_smoothings=(0.0)
cycle=10
dataset_name="wnut17"
dataset_path="wnut17"
dataset_method_name="switch"
method="switch"
model_name="flan-t5-base"

# FacebookAI/xlm-roberta-base
# google/t5gemma-b-b-ul2

# fewnerd batch 112
# conll2003 batch 384 large 128
# wnut17 batch 384
# jnlpba batch 256 

# warmup/logging steps
# wnut17 10/2
# jnlpba 60/20
# mit_movie 10/4

index=0
while [ $index -lt 1 ]
do
    echo "Training with learning rate: ${learning_rates[$index]}"
    dropout_percentage=$(echo "scale=3; ${dropout_rates[$index]}*100" | bc)
    dropout_percentage=$(printf "%.0f" "$dropout_percentage")
    smoothing_percentage=$(echo "scale=3; ${label_smoothings[$index]}*100" | bc)
    smoothing_percentage=$(printf "%.0f" "$smoothing_percentage")

    python /workspace/encoder_binary/encoder_classification_train.py \
        --model_checkpoint google/flan-t5-base \
        --output_dir /workspace/model_dir/classification/${model_name}/${dataset_name}/encoder-${method}-ner-custom-class_weight-drop${dropout_percentage}-smoothing${smoothing_percentage}-cycle${cycle}-lr${learning_rates[$index]}-cosine_restart \
        --custom_model \
        --dynamic_class_weights \
        --weight_decay 0.001 \
        --label_smoothing ${label_smoothings[$index]} \
        --gradient_accumulation_steps 1 \
        --batch_size 384 \
        --dropout_rate ${dropout_rates[$index]} \
        --num_cycles $cycle \
        --learning_rate ${learning_rates[$index]} \
        --num_epochs 20 \
        --warmup_steps 10 \
        --logging_steps 2 \
        --early_stopping_patience 5 \
        --dataset_name ${dataset_method_name} \
        --train_file /workspace/datas/${dataset_path}/train.${method}.csv \
        --validation_file /workspace/datas/${dataset_path}/dev.${method}.csv

    index=`expr $index + 1`
done