model_checkpoint_name=("ner-inerd2_conll2003-conll2003_contrastive_batch64-fp32-w1e3-lr1e4-seed_42")
model_name=("flan-t5-base")
model_train_phrase=("second")
dataset_names=("conll2003")
dataset_paths=("/workspace/datas/conll2003/testb.inerd2.csv")
prompt_types=("inerd2")
index=0
while [ $index -lt 1 ]
do
    model_checkpoint_path="/workspace/model_dir/generation/${model_name[$index]}/${model_train_phrase[$index]}/${model_checkpoint_name[$index]}/final_model"

    python /workspace/ner_generation/generation_inference.py \
        --model_checkpoint ${model_checkpoint_path} \
        --output_file /workspace/datas/generated/${model_name[$index]}-${model_train_phrase[$index]}-${model_checkpoint_name[$index]}-tuned-${dataset_names[$index]}.csv \
        --batch_size 32 \
        --dataset_name ${dataset_names[$index]} \
        --data_file ${dataset_paths[$index]} \
        --prompt_type ${prompt_types[$index]}
    
    index=`expr $index + 1`
done
