dataset_path="ace05"
model_name="gemma-3-270m-it"

dataset_methods=("ace05")
trainset_paths=("/workspace/datas/${dataset_path}/train.inerd.csv")
validset_paths=("/workspace/datas/${dataset_path}/dev.inerd.csv")

#--encoder_weight /workspace/model_dir/classification/${model_name}/encoder-switch-conll2003-ner-lr2e-5-cosine_restart/final_model \
# t5-base fewnerd batch 32

index=0
while [ $index -lt 1 ]
do
    python /workspace/ner_generation/generation_decoder_train.py \
        --model_checkpoint google/gemma-3-270m-it \
        --output_dir /workspace/model_dir/generation/${model_name}/ner-inerd_${dataset_methods[$index]}-fp32-w1e3-lr8e5 \
        --train_epochs 20 \
        --weight_decay 0.001 \
        --warmup_steps 20 \
        --logging_steps 5 \
        --batch_size 6 \
        --gradient_accumulation_steps 16 \
        --learning_rate 8e-5 \
        --dataset_name ${dataset_methods[$index]} \
        --token_setting inerd \
        --train_file ${trainset_paths[$index]} \
        --validation_file ${validset_paths[$index]}

    index=`expr $index + 1`
done