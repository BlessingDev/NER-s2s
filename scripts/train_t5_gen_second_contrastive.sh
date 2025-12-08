model_name="flan-t5-base"
model_checkpoints=("/workspace/model_dir/generation/flan-t5-base/first/ner-inerd2_conll2003-conll2003-fp32-w1e3-lr1e4-seed_42/final_model")

dataset_name=("conll2003_contrastive_batch64")
dataset_methods=("conll2003")
inerd_method=("inerd2")
trainset_paths=("/workspace/datas/conll2003/train.inerd2.contrastive.csv")
validset_paths=("/workspace/datas/conll2003/dev.inerd2.contrastive.csv")
batch_size=(64)
gradient_accumulation_steps=(2)
warmup_steps=(130)
logging_steps=(20)
seed=(42)
#--encoder_weight /workspace/model_dir/classification/${model_name}/encoder-switch-conll2003-ner-lr2e-5-cosine_restart/final_model \

# conll 130/20  batch 128까지 가능
# jnlpba 
# wnut17 
# mit_restaurant 
# ontonotes5 

# ace05 
# genia 

index=0
while [ $index -lt 1 ]
do
    python /workspace/ner_generation/generation_s2s_train_second.py \
        --model_checkpoint ${model_checkpoints[$index]} \
        --output_dir /workspace/model_dir/generation/${model_name}/second/ner-${inerd_method[$index]}_${dataset_methods[$index]}-${dataset_name[$index]}-fp32-w1e3-lr1e4-seed_${seed[$index]} \
        --train_method contrastive \
        --train_epochs 20 \
        --num_cycles 10 \
        --weight_decay 0.001 \
        --contrastive_temperature 0.5 \
        --contrastive_lambda 0.5 \
        --warmup_steps ${warmup_steps[$index]} \
        --logging_steps ${logging_steps[$index]} \
        --batch_size ${batch_size[$index]} \
        --gradient_accumulation_steps ${gradient_accumulation_steps[$index]} \
        --learning_rate 1e-4 \
        --dataset_name ${dataset_methods[$index]} \
        --token_setting inerd \
        --train_file ${trainset_paths[$index]} \
        --validation_file ${validset_paths[$index]} \
        --seed ${seed[$index]}

    index=`expr $index + 1`
done

echo "Start Reserved Inference"
bash /workspace/scripts/inference_gen.sh