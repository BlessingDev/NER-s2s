#dataset_path=("fewnerd")
model_name="flan-t5-base"

dataset_name=("conll2003_masktoken")
dataset_methods=("conll2003")
inerd_method=("inerd2")
trainset_paths=("/workspace/datas/conll2003/train.inerd2.csv")
validset_paths=("/workspace/datas/conll2003/dev.inerd2.csv")
batch_size=(128)
gradient_accumulation_steps=(1)
warmup_steps=(40)
logging_steps=(20)
seed=(42)
#--encoder_weight /workspace/model_dir/classification/${model_name}/encoder-switch-conll2003-ner-lr2e-5-cosine_restart/final_model \
# t5-base fewnerd batch 32

# fewnerd 500/50 batch 32
# conll 40/20  batch 128까지 가능
# jnlpba 200/50 curriculum2: 240/70 batch 64
# wnut17 10/5
# mit_restaurant 20/12 batch 128
# ontonotes5 200/50 batch 48

# ace05 40/10 curriculum: 100/30 batch 128 
# genia 50/10 curriculum: 100/30 batch 64

index=0
while [ $index -lt 1 ]
do
    python /workspace/ner_generation/generation_s2s_train_first.py \
        --model_checkpoint google/${model_name} \
        --output_dir /workspace/model_dir/generation/${model_name}/first/ner-${inerd_method[$index]}_${dataset_methods[$index]}-${dataset_name[$index]}-fp32-w1e2-lr1e4-seed_${seed[$index]} \
        --train_epochs 20 \
        --num_cycles 10 \
        --weight_decay 0.01 \
        --warmup_steps ${warmup_steps[$index]} \
        --logging_steps ${logging_steps[$index]} \
        --batch_size ${batch_size[$index]} \
        --gradient_accumulation_steps ${gradient_accumulation_steps[$index]} \
        --learning_rate 1e-4 \
        --dataset_name ${dataset_methods[$index]} \
        --token_setting inerd \
        --prompt_setting inerd \
        --train_file ${trainset_paths[$index]} \
        --validation_file ${validset_paths[$index]} \
        --seed ${seed[$index]}

    index=`expr $index + 1`
done

echo "Start Reserved Inference"
bash /workspace/scripts/inference_gen.sh